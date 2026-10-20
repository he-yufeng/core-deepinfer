"""A random, untrained causal encoder with shared-KV and local-attention consumers."""

import hashlib
from dataclasses import dataclass

import numpy as np


def positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def norm(x):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + 1e-6)


def attention(q, k, v, visible):
    scores = (q @ k.T) / np.sqrt(q.shape[-1])
    scores = np.where(visible, scores, -np.inf)
    maximum = np.max(scores, axis=-1, keepdims=True)
    maximum = np.where(np.isfinite(maximum), maximum, 0.0)
    weights = np.exp(scores - maximum)
    weights /= np.maximum(weights.sum(axis=-1, keepdims=True), 1e-300)
    return weights @ v


@dataclass
class EncoderState:
    owner: str
    tokens: tuple[int, ...]
    encoder_keys: np.ndarray
    encoder_values: np.ndarray
    source: np.ndarray
    shared_keys: np.ndarray
    shared_values: np.ndarray


@dataclass
class State(EncoderState):
    decoder_keys: tuple[np.ndarray, ...]
    decoder_values: tuple[np.ndarray, ...]


@dataclass
class Forward:
    logits: np.ndarray
    state: State
    positions: np.ndarray
    stats: dict[str, int]


class TinyModel:
    """Single-head float64 reference. All weights are fixed random matrices."""

    def __init__(self, *, vocab_size=32, width=16, depth=3, window=4, seed=0):
        self.vocab_size = positive_int(vocab_size, "vocab_size")
        self.width = positive_int(width, "width")
        self.depth = positive_int(depth, "depth")
        self.window = positive_int(window, "window")
        rng = np.random.default_rng(seed)

        def matrix():
            return rng.normal(0, 1 / np.sqrt(width), (width, width))

        self.embedding = rng.normal(0, 0.3, (vocab_size, width))
        self.encoder = tuple(matrix() for _ in range(6))
        self.shared = (matrix(), matrix())
        self.layers = [tuple(matrix() for _ in range(8)) for _ in range(depth)]
        self.head = rng.normal(0, 1 / np.sqrt(width), (width, vocab_size))
        digest = hashlib.sha256(f"{vocab_size}:{width}:{depth}:{window}".encode())
        for value in [
            self.embedding,
            *self.encoder,
            *self.shared,
            *(m for layer in self.layers for m in layer),
            self.head,
        ]:
            digest.update(value.tobytes())
        self.identity = digest.hexdigest()

    def tokens(self, tokens):
        values = list(tokens)
        if not values or any(
            isinstance(t, (bool, np.bool_))
            or not isinstance(t, (int, np.integer))
            or not 0 <= t < self.vocab_size
            for t in values
        ):
            raise ValueError("tokens must be a nonempty sequence of in-vocabulary integers")
        return tuple(int(t) for t in values)

    def embed(self, tokens, positions):
        frequency = np.power(10000.0, -np.arange(self.width) / self.width)
        return self.embedding[list(tokens)] + 0.1 * np.sin(positions[:, None] * frequency)

    def prefill(self, tokens, *, mode="exact", keep=None):
        tokens = self.tokens(tokens)
        if mode not in ("exact", "mask", "compact"):
            raise ValueError("mode must be exact, mask or compact")
        if mode != "exact":
            keep = positive_int(keep, "keep")
            if keep < self.window:
                raise ValueError("keep must cover at least one local window")
        n = len(tokens)
        positions = np.arange(n)
        x = self.embed(tokens, positions)
        wq, wk, wv, wo, f1, f2 = self.encoder
        base = norm(x)
        keys, values = base @ wk, base @ wv
        x = x + 0.25 * attention(base @ wq, keys, values, positions[None, :] <= positions[:, None]) @ wo
        source = x + 0.2 * np.tanh(norm(x) @ f1) @ f2
        encoder = EncoderState(
            self.identity,
            tokens,
            keys,
            values,
            source,
            source @ self.shared[0],
            source @ self.shared[1],
        )
        return self._decode(encoder, mode=mode, keep=keep, encoded_rows=n)

    def replay(self, encoder, *, mode="exact", keep=None):
        return self._decode(encoder, mode=mode, keep=keep, encoded_rows=0)

    def _decode(self, encoder, *, mode, keep, encoded_rows):
        if not isinstance(encoder, EncoderState) or encoder.owner != self.identity:
            raise ValueError("encoder source belongs to different model parameters or layout")
        tokens = self.tokens(encoder.tokens)
        n = len(tokens)
        keys, values, source = encoder.encoder_keys, encoder.encoder_values, encoder.source
        shared_k, shared_v = encoder.shared_keys, encoder.shared_values
        if any(
            value.shape != (n, self.width) or not np.isfinite(value).all()
            for value in (keys, values, source, shared_k, shared_v)
        ):
            raise ValueError("encoder arrays must match the token prefix and model width")
        if mode not in ("exact", "mask", "compact"):
            raise ValueError("mode must be exact, mask or compact")
        if mode != "exact":
            keep = positive_int(keep, "keep")
            if keep < self.window:
                raise ValueError("keep must cover at least one local window")
        positions = np.arange(n)
        lower = 0 if mode == "exact" else max(0, n - keep)
        live = positions[positions >= lower] if mode == "compact" else positions
        x = source[live].copy()
        local_mask = (
            (live[None, :] <= live[:, None])
            & (live[None, :] >= live[:, None] - self.window + 1)
            & (live[None, :] >= lower)
        )
        cross_mask = positions[None, :] <= live[:, None]
        decoder_k, decoder_v = [], []
        for wq, wk, wv, wo, cq, co, f1, f2 in self.layers:
            base = norm(x)
            k, v = base @ wk, base @ wv
            local = attention(base @ wq, k, v, local_mask) @ wo
            cross = attention(base @ cq, shared_k, shared_v, cross_mask) @ co
            x = x + 0.25 * (local + cross)
            x = x + 0.2 * np.tanh(norm(x) @ f1) @ f2
            decoder_k.append(k[-self.window :].copy())
            decoder_v.append(v[-self.window :].copy())
        state = State(
            self.identity,
            tokens,
            keys,
            values,
            source,
            shared_k,
            shared_v,
            tuple(decoder_k),
            tuple(decoder_v),
        )
        return Forward(
            norm(x) @ self.head,
            state,
            live,
            {"encoder_rows": encoded_rows, "consumer_rows": len(live) * self.depth},
        )

    def step(self, state, token):
        if state.owner != self.identity:
            raise ValueError("state belongs to different model parameters or layout")
        token = self.tokens([token])[0]
        pos = len(state.tokens)
        x = self.embed([token], np.array([pos]))
        wq, wk, wv, wo, f1, f2 = self.encoder
        base = norm(x)
        keys = np.concatenate([state.encoder_keys, base @ wk])
        values = np.concatenate([state.encoder_values, base @ wv])
        x = x + 0.25 * attention(base @ wq, keys, values, np.ones((1, pos + 1), dtype=bool)) @ wo
        new_source = x + 0.2 * np.tanh(norm(x) @ f1) @ f2
        source = np.concatenate([state.source, new_source])
        shared_k = np.concatenate([state.shared_keys, new_source @ self.shared[0]])
        shared_v = np.concatenate([state.shared_values, new_source @ self.shared[1]])
        x = source[-1:].copy()
        decoder_k, decoder_v = [], []
        for index, (wq, wk, wv, wo, cq, co, f1, f2) in enumerate(self.layers):
            base = norm(x)
            k = np.concatenate([state.decoder_keys[index], base @ wk])[-self.window :]
            v = np.concatenate([state.decoder_values[index], base @ wv])[-self.window :]
            local = attention(base @ wq, k, v, np.ones((1, len(k)), dtype=bool)) @ wo
            cross = attention(base @ cq, shared_k, shared_v, np.ones((1, len(source)), dtype=bool)) @ co
            x = x + 0.25 * (local + cross)
            x = x + 0.2 * np.tanh(norm(x) @ f1) @ f2
            decoder_k.append(k)
            decoder_v.append(v)
        state = State(
            self.identity,
            (*state.tokens, token),
            keys,
            values,
            source,
            shared_k,
            shared_v,
            tuple(decoder_k),
            tuple(decoder_v),
        )
        return Forward(
            norm(x) @ self.head, state, np.array([pos]), {"encoder_rows": 1, "consumer_rows": self.depth}
        )
