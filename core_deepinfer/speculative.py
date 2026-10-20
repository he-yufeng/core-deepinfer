"""Greedy speculative semantics using a recomputing oracle, not a faster kernel."""

from dataclasses import dataclass

import numpy as np

from .model import State, positive_int


@dataclass(frozen=True)
class Verification:
    tokens: tuple[int, ...]
    accepted: int


@dataclass
class Generation:
    tokens: tuple[int, ...]
    state: State
    accepted_drafts: int = 0
    draft_tokens: int = 0


def verify_greedy(logits, drafts, *, eos_id=None):
    drafts = tuple(drafts)
    logits = np.asarray(logits)
    if logits.ndim != 2 or logits.shape[0] != len(drafts) + 1 or not np.isfinite(logits).all():
        raise ValueError("verification needs one target distribution per draft plus the bonus")
    if any(type(t) is not int or not 0 <= t < logits.shape[1] for t in drafts):
        raise ValueError("draft token is outside the target vocabulary")
    predictions = logits.argmax(axis=1)
    accepted = 0
    committed = []
    for index, draft in enumerate(drafts):
        prediction = int(predictions[index])
        committed.append(prediction)
        if draft == prediction:
            accepted += 1
        if draft != prediction or prediction == eos_id:
            return Verification(tuple(committed), accepted)
    committed.append(int(predictions[-1]))
    return Verification(tuple(committed), accepted)


def token_limit(value):
    if type(value) is not int or value < 0:
        raise ValueError("max_tokens must be a nonnegative integer")
    return value


def generate(model, prompt, *, max_tokens=16, eos_id=None):
    token_limit(max_tokens)
    forward = model.prefill(prompt)
    emitted = []
    for _ in range(max_tokens):
        token = int(forward.logits[-1].argmax())
        emitted.append(token)
        forward = model.step(forward.state, token)
        if token == eos_id:
            break
    return Generation(tuple(emitted), forward.state)


def generate_speculative(target, draft, prompt, *, max_tokens=16, draft_width=5, eos_id=None):
    token_limit(max_tokens)
    draft_width = positive_int(draft_width, "draft_width")
    if target.vocab_size != draft.vocab_size:
        raise ValueError("target and draft must use the same vocabulary")
    prefix = target.tokens(prompt)
    emitted = []
    accepted = proposed = 0
    while len(emitted) < max_tokens:
        context = (*prefix, *emitted)
        width = min(draft_width, max_tokens - len(emitted))
        drafts = generate(draft, context, max_tokens=width).tokens
        proposed += len(drafts)
        verified = target.prefill((*context, *drafts))
        start = len(context) - 1
        decision = verify_greedy(verified.logits[start:], drafts, eos_id=eos_id)
        remaining = max_tokens - len(emitted)
        committed = decision.tokens[:remaining]
        accepted += min(decision.accepted, len(committed))
        emitted.extend(committed)
        if committed[-1] == eos_id:
            break
    # Rebuild only the committed branch; proposed suffixes never leak into it.
    state = target.prefill((*prefix, *emitted)).state
    return Generation(tuple(emitted), state, accepted, proposed)
