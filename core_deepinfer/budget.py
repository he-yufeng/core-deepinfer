"""Expected-output budget search with caller-supplied, not measured, costs."""

import math
from bisect import bisect_left
from dataclasses import dataclass

import numpy as np

from .model import positive_int


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


def _limits(rates, draft_limits):
    if draft_limits is None:
        return (rates.shape[1],) * len(rates)
    limits = tuple(draft_limits)
    if len(limits) != len(rates) or any(
        isinstance(n, (bool, np.bool_))
        or not isinstance(n, (int, np.integer))
        or not 0 <= n <= rates.shape[1]
        for n in limits
    ):
        raise ValueError("draft_limits must match the available positions of every request")
    return tuple(int(n) for n in limits)


def ranking(rates, draft_limits=None):
    limits = _limits(rates, draft_limits)
    return sorted(
        (-float(rate), request, position)
        for request, row in enumerate(rates)
        for position, rate in enumerate(row[: limits[request]])
    )


def reallocate(total, confidence, *, draft_limits=None):
    rates = survival_probabilities(confidence)
    limits = _limits(rates, draft_limits)
    requests = len(rates)
    if type(total) is not int or not requests <= total <= requests + sum(limits):
        raise ValueError("budget must fit the main positions and available drafts")
    lengths = [1] * requests
    for _, request, _ in ranking(rates, limits)[: total - requests]:
        lengths[request] += 1
    return tuple(lengths)


def choose_budget(confidence, curve, *, sampler_ms=0.0, slope_ms=0.0, draft_limits=None, max_budget=None):
    rates = survival_probabilities(confidence)
    limits = _limits(rates, draft_limits)
    requests = len(rates)
    cap = requests + sum(limits)
    if max_budget is not None:
        cap = min(cap, positive_int(max_budget, "max_budget"))
        if cap < requests:
            raise ValueError("max_budget cannot fit all main token positions")
    if any(not math.isfinite(x) or x < 0 for x in (sampler_ms, slope_ms)):
        raise ValueError("runtime costs must be finite and nonnegative")
    ranked = ranking(rates, limits)
    if not curve:
        return Budget(
            cap,
            reallocate(cap, confidence, draft_limits=limits),
            None,
            float(requests + sum(-entry[0] for entry in ranked[: cap - requests])),
            None,
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
    candidates = [0.0, *(-entry[0] for entry in ranked)]
    best = None
    expected = float(requests)
    for added, benefit in enumerate(candidates):
        total = requests + added
        if total > cap:
            break
        expected += benefit
        graph_index = bisect_left(sizes, total)
        if graph_index == len(sizes):
            break
        graph_size, baseline = curve[graph_index]
        cost = baseline + slope_ms * graph_size + sampler_ms
        candidate = Budget(
            total,
            reallocate(total, confidence, draft_limits=limits),
            graph_size,
            expected,
            cost,
        )
        if best is None or expected / cost > best.expected_tokens / best.estimated_ms:
            best = candidate
    return best
