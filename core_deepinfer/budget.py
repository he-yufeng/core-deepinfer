"""Expected-output budget search with caller-supplied, not measured, costs."""

import math
from bisect import bisect_left
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Budget:
    total_tokens: int
    lengths: tuple[int, ...]
    graph_size: int | None
    expected_tokens: float
    estimated_ms: float | None


def survival_probabilities(confidence):
    values = np.asarray(confidence, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("confidence must have shape [requests, draft_positions]")
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("confidence must be finite and lie in [0, 1]")
    return np.cumprod(values, axis=1)


def ranking(rates):
    return sorted(
        (
            (-float(rate), request, position)
            for request, row in enumerate(rates)
            for position, rate in enumerate(row)
        )
    )


def reallocate(total, confidence):
    rates = survival_probabilities(confidence)
    requests = len(rates)
    if type(total) is not int or not requests <= total <= requests + rates.size:
        raise ValueError("budget must fit the main positions and available drafts")
    lengths = [1] * requests
    for _, request, _ in ranking(rates)[: total - requests]:
        lengths[request] += 1
    return tuple(lengths)


def choose_budget(confidence, curve, *, sampler_ms=0.0, slope_ms=0.0):
    rates = survival_probabilities(confidence)
    requests = len(rates)
    if any(not math.isfinite(x) or x < 0 for x in (sampler_ms, slope_ms)):
        raise ValueError("runtime costs must be finite and nonnegative")
    if not curve:
        return Budget(
            requests + rates.size, (1 + rates.shape[1],) * requests, None, float(requests + rates.sum()), None
        )
    sizes = []
    for size, cost in curve:
        if type(size) is not int or size < 1 or (sizes and size <= sizes[-1]):
            raise ValueError("graph sizes must be positive and strictly increasing")
        if not math.isfinite(cost) or cost <= 0:
            raise ValueError("graph costs must be finite and positive")
        sizes.append(size)
    if sizes[-1] < requests:
        raise ValueError("no graph can fit the main token positions")
    candidates = [0.0, *(-entry[0] for entry in ranking(rates))]
    best = None
    expected = float(requests)
    for added, benefit in enumerate(candidates):
        expected += benefit
        total = requests + added
        graph_index = bisect_left(sizes, total)
        if graph_index == len(sizes):
            break
        graph_size, baseline = curve[graph_index]
        cost = baseline + slope_ms * graph_size + sampler_ms
        candidate = Budget(total, reallocate(total, confidence), graph_size, expected, cost)
        if best is None or expected / cost > best.expected_tokens / best.estimated_ms:
            best = candidate
    return best
