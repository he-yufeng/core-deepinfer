from itertools import product

import numpy as np
import pytest

from core_deepinfer.budget import choose_budget, reallocate, survival_probabilities


def test_confidence_is_multiplied_over_the_prefix():
    np.testing.assert_allclose(survival_probabilities([[0.8] * 5]), [[0.8**n for n in range(1, 6)]])


@pytest.mark.parametrize("seed", [0, 1, 5, 8])
def test_budget_search_matches_exhaustive_length_allocations(seed):
    confidence = np.random.default_rng(seed).uniform(0.1, 0.99, (2, 3))
    curve = [(2, 1.0), (4, 1.4), (8, 4.0)]
    result = choose_budget(confidence, curve, sampler_ms=0.3)
    rates = survival_probabilities(confidence)
    scores = []
    for depths in product(range(4), repeat=2):
        total = 2 + sum(depths)
        graph, cost = next(point for point in curve if point[0] >= total)
        expected = 2 + sum(rates[i, :depth].sum() for i, depth in enumerate(depths))
        scores.append(expected / (cost + 0.3))
    assert result.expected_tokens / result.estimated_ms == pytest.approx(max(scores))
    assert sum(result.lengths) == result.total_tokens
    assert result.graph_size >= result.total_tokens


def test_padding_is_not_given_to_requests_as_verification_work():
    result = choose_budget([[0.95, 0.01]], [(1, 1.0), (4, 1.5)])
    assert result.total_tokens == 3
    assert result.graph_size == 4
    assert result.lengths == (3,)


def test_new_confidence_redistributes_without_changing_total_budget():
    assert reallocate(5, [[0.99, 0.99, 0.99], [0.1, 0.1, 0.1]]) == (4, 1)
    assert reallocate(5, [[0.1, 0.1, 0.1], [0.99, 0.99, 0.99]]) == (1, 4)


def test_no_curve_keeps_every_draft():
    result = choose_budget([[0.1, 0.2], [0.3, 0.4]], None)
    assert result.lengths == (3, 3)
    assert result.estimated_ms is None
    assert result.graph_size is None


@pytest.mark.parametrize("confidence", [[[float("nan")]], [[float("inf")]], [[-0.1]], [[1.1]], [], [0.5]])
def test_invalid_confidence_is_rejected(confidence):
    with pytest.raises(ValueError):
        choose_budget(confidence, [(4, 1.0)])


@pytest.mark.parametrize("curve", [[(1, 0)], [(2, float("nan"))], [(4, 1), (2, 2)], [(2, 1), (2, 2)]])
def test_invalid_cost_curves_are_rejected(curve):
    with pytest.raises(ValueError):
        choose_budget([[0.5]], curve)


@pytest.mark.parametrize("total", [0, 5, 1.5])
def test_reallocation_cannot_invent_or_drop_budget(total):
    with pytest.raises(ValueError):
        reallocate(total, [[0.5, 0.5]])
