"""Bounded replay experiments; exact recovery is a separate reference graph."""

from dataclasses import dataclass

import numpy as np

from .model import TinyModel, positive_int


@dataclass(frozen=True)
class ChunkPlan:
    positions: tuple[int, ...]
    ring_slots: tuple[int, ...]


def chunk_plan(total, start, length, *, keep, window, rollback=0):
    total = positive_int(total, "total")
    length = positive_int(length, "length")
    keep = positive_int(keep, "keep")
    window = positive_int(window, "window")
    if type(start) is not int or start < 0 or start + length > total:
        raise ValueError("chunk must lie inside the sequence")
    if type(rollback) is not int or rollback < 0 or keep < window + rollback:
        raise ValueError("keep must cover window and rollback")
    clipped = max(0, min(total - keep - start, length - 1))
    positions = tuple(range(start + clipped, start + length))
    ring_size = window + rollback
    slots = tuple(
        p % ring_size if p >= total - keep and start + length - p <= ring_size else -1 for p in positions
    )
    return ChunkPlan(positions, slots)


def replay_experiment(*, length=32, keep=8, seed=7):
    length = positive_int(length, "length")
    model = TinyModel(seed=seed, window=4)
    tokens = [int(t) for t in np.random.default_rng(seed + 1).integers(0, model.vocab_size, length)]
    exact = model.prefill(tokens)
    mask = model.prefill(tokens, mode="mask", keep=keep)
    compact = model.prefill(tokens, mode="compact", keep=keep)
    return {
        "model_weights": "random, untrained",
        "length": length,
        "keep": keep,
        "mask_consumer_rows": mask.stats["consumer_rows"],
        "compact_consumer_rows": compact.stats["consumer_rows"],
        "full_source_rows": len(compact.state.source),
        "mask_compact_max_error": float(np.max(np.abs(mask.logits[compact.positions] - compact.logits))),
        "bounded_exact_max_error": float(np.max(np.abs(exact.logits[compact.positions] - compact.logits))),
        "scope": "Float64 toy-model arithmetic and row counts, not DeepSeek quality or a speed benchmark",
    }
