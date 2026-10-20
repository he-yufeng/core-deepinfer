"""Incremental execution with real local KV rings and reversible speculative writes."""

from copy import deepcopy
from dataclasses import dataclass
from uuid import uuid4

import numpy as np

from .cache import RingSnapshot, StateRing
from .model import EncoderState, Forward, State
from .speculative import token_limit, verify_greedy


@dataclass
class CachedState:
    state: State
    logits: np.ndarray


@dataclass
class Checkpoint:
    owner: str
    encoder: EncoderState
    keys: RingSnapshot
    values: RingSnapshot
    logits: np.ndarray


@dataclass
class Proposal:
    tokens: tuple[int, ...]
    confidence: tuple[float, ...]
    checkpoints: tuple[Checkpoint, ...]


@dataclass(frozen=True)
class Verified:
    tokens: tuple[int, ...]
    accepted: int
    verified: int
    discarded: int


def encoder_state(state):
    return EncoderState(
        state.owner,
        state.tokens,
        state.encoder_keys,
        state.encoder_values,
        state.source,
        state.shared_keys,
        state.shared_values,
    )


class Runner:
    def __init__(self, model, forward):
        self.model = model
        state = forward.state
        n = len(state.tokens)
        local_rows = min(n, model.window)
        if state.owner != model.identity or model.tokens(state.tokens) != state.tokens:
            raise ValueError("cached state belongs to another model or token prefix")
        if any(
            value.shape != (n, model.width) or not np.isfinite(value).all()
            for value in (
                state.encoder_keys,
                state.encoder_values,
                state.source,
                state.shared_keys,
                state.shared_values,
            )
        ):
            raise ValueError("cached encoder arrays have an invalid shape or value")
        if (
            len(state.decoder_keys) != model.depth
            or len(state.decoder_values) != model.depth
            or any(
                value.shape != (local_rows, model.width) or not np.isfinite(value).all()
                for value in (*state.decoder_keys, *state.decoder_values)
            )
        ):
            raise ValueError("cached local KV does not match the model layout")
        logits = np.asarray(forward.logits, dtype=np.float64)
        if logits.ndim != 2 or logits.shape[1] != model.vocab_size or not np.isfinite(logits).all():
            raise ValueError("runner needs finite next-token logits")
        self.encoder = deepcopy(encoder_state(state))
        self.logits = logits[-1].copy()
        self.keys = StateRing(model.window, model.depth * model.width)
        self.values = StateRing(model.window, model.depth * model.width)
        self.keys.next_position = self.values.next_position = n - local_rows
        for row in range(local_rows):
            self.keys.append(np.concatenate([layer[row] for layer in state.decoder_keys]))
            self.values.append(np.concatenate([layer[row] for layer in state.decoder_values]))
        self.owner = uuid4().hex
        self.steps = 0
        self.stats = dict(forward.stats)

    @classmethod
    def from_cached(cls, model, cached):
        forward = Forward(
            cached.logits[None, :],
            cached.state,
            np.array([len(cached.state.tokens) - 1]),
            {"encoder_rows": 0, "consumer_rows": 0},
        )
        return cls(model, forward)

    @property
    def tokens(self):
        return self.encoder.tokens

    @property
    def state(self):
        n = len(self.tokens)
        positions = range(max(0, n - self.model.window), n)
        keys = np.stack([self.keys.read(i) for i in positions]).reshape(
            -1, self.model.depth, self.model.width
        )
        values = np.stack([self.values.read(i) for i in positions]).reshape(
            -1, self.model.depth, self.model.width
        )
        e = self.encoder
        return State(
            e.owner,
            e.tokens,
            e.encoder_keys.copy(),
            e.encoder_values.copy(),
            e.source.copy(),
            e.shared_keys.copy(),
            e.shared_values.copy(),
            tuple(keys[:, i, :].copy() for i in range(self.model.depth)),
            tuple(values[:, i, :].copy() for i in range(self.model.depth)),
        )

    def capture(self):
        return CachedState(self.state, self.logits.copy())

    def checkpoint(self):
        return Checkpoint(
            self.owner,
            deepcopy(self.encoder),
            self.keys.checkpoint(),
            self.values.checkpoint(),
            self.logits.copy(),
        )

    def rollback(self, checkpoint):
        if checkpoint.owner != self.owner:
            raise ValueError("checkpoint belongs to another runner")
        self.keys.rollback(checkpoint.keys)
        self.values.rollback(checkpoint.values)
        self.encoder = deepcopy(checkpoint.encoder)
        self.logits = checkpoint.logits.copy()

    def advance(self, token):
        forward = self.model.step(self.state, token)
        keys = np.concatenate([layer[-1] for layer in forward.state.decoder_keys])
        values = np.concatenate([layer[-1] for layer in forward.state.decoder_values])
        expected_width = self.model.depth * self.model.width
        if (
            keys.shape != (expected_width,)
            or values.shape != (expected_width,)
            or forward.logits.shape != (1, self.model.vocab_size)
        ):
            raise ValueError("model produced logits or KV with the wrong shape")
        if not all(np.isfinite(value).all() for value in (forward.logits, keys, values)):
            raise ValueError("model produced nonfinite logits or KV")
        self.keys.append(keys)
        self.values.append(values)
        self.encoder = encoder_state(forward.state)
        self.logits = forward.logits[-1].copy()
        self.steps += 1
        for key, count in forward.stats.items():
            self.stats[key] = self.stats.get(key, 0) + count
        return self.logits.copy()

    def propose(self, width, *, eos_id=None):
        token_limit(width)
        checkpoints = [self.checkpoint()]
        tokens, confidence = [], []
        try:
            for _ in range(width):
                token = int(self.logits.argmax())
                weights = np.exp(self.logits - self.logits.max())
                confidence.append(float(weights[token] / weights.sum()))
                tokens.append(token)
                self.advance(token)
                checkpoints.append(self.checkpoint())
                if token == eos_id:
                    break
        except Exception:
            self.rollback(checkpoints[0])
            raise
        return Proposal(tuple(tokens), tuple(confidence), tuple(checkpoints))

    def verify(self, drafts, *, max_tokens, eos_id=None):
        token_limit(max_tokens)
        if max_tokens == 0:
            raise ValueError("verification must have room for at least one committed token")
        drafts = tuple(drafts)
        if drafts:
            drafts = self.model.tokens(drafts)
        if len(drafts) >= max_tokens:
            raise ValueError("drafts must leave one output position for replacement or bonus")
        checkpoints = [self.checkpoint()]
        logits = [self.logits.copy()]
        try:
            for token in drafts:
                logits.append(self.advance(token))
                checkpoints.append(self.checkpoint())
            decision = verify_greedy(np.stack(logits), drafts, eos_id=eos_id)
            accepted = decision.accepted
            discarded = len(drafts) - accepted
            if discarded:
                self.rollback(checkpoints[accepted])
            for token in decision.tokens[accepted:]:
                self.advance(token)
        except Exception:
            self.rollback(checkpoints[0])
            raise
        return Verified(decision.tokens, accepted, len(drafts), discarded)
