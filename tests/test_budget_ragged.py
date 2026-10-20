from itertools import product

import numpy as np
import pytest

from core_deepinfer.budget import choose_budget, reallocate, survival_probabilities


@pytest.mark.parametrize("seed", [0, 7, 19])
def test_ragged_budget_matches_an_exhaustive_oracle_without_inventing_padding_drafts(seed):
    confidence = np.random.default_rng(seed).uniform(0.1, 0.99, (3, 4))
    limits = [4, 1, 0]
    curve = [(3, 1.0), (4, 1.1), (8, 1.9)]
    result = choose_budget(confidence, curve, draft_limits=limits, max_budget=6)
    rates = survival_probabilities(confidence)
    scores = []
    for lengths in product(*[range(n + 1) for n in limits]):
        total = len(limits) + sum(lengths)
        if total > 6:
            continue
        _, cost = next(point for point in curve if point[0] >= total)
        expected = len(limits) + sum(rates[row, :length].sum() for row, length in enumerate(lengths))
        scores.append(expected / cost)
    assert result.expected_tokens / result.estimated_ms == pytest.approx(max(scores))
    assert all(length <= limit + 1 for length, limit in zip(result.lengths, limits, strict=True))
    assert sum(result.lengths) == result.total_tokens <= 6


def test_no_curve_still_respects_available_drafts_and_the_batch_cap():
    result = choose_budget([[0.9, 0.9, 0.9], [0.8, 0.8, 0.8]], None, draft_limits=[0, 3], max_budget=4)
    assert result.total_tokens == 4
    assert result.lengths == (1, 3)
    assert result.graph_size is None
    assert result.estimated_ms is None


def test_zero_draft_batch_has_one_main_position_per_request():
    result = choose_budget(np.empty((3, 0)), None, max_budget=3)
    assert result.lengths == (1, 1, 1)
    assert result.total_tokens == 3
    assert reallocate(3, np.empty((3, 0)), draft_limits=[0, 0, 0]) == (1, 1, 1)


@pytest.mark.parametrize("limits", [[1], [3, 1], [-1, 1], [True, 1], [1.5, 1]])
def test_invalid_availability_cannot_reach_the_planner(limits):
    with pytest.raises(ValueError):
        choose_budget([[0.5, 0.5], [0.5, 0.5]], None, draft_limits=limits)


def test_reallocation_cannot_spend_budget_on_unavailable_positions():
    with pytest.raises(ValueError):
        reallocate(4, [[0.9, 0.9], [0.8, 0.8]], draft_limits=[1, 0])
    with pytest.raises(ValueError, match="main token"):
        choose_budget([[0.9], [0.8]], None, max_budget=1)
