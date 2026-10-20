import numpy as np
import pytest

from core_deepinfer.model import TinyModel
from core_deepinfer.speculative import generate, generate_speculative, verify_greedy


@pytest.mark.parametrize("seed", [0, 3, 7])
@pytest.mark.parametrize("width", [1, 3, 5])
def test_speculative_output_and_committed_state_match_greedy(seed, width):
    target = TinyModel(seed=seed, window=4)
    draft = TinyModel(seed=seed + 1, window=4, depth=1)
    plain = generate(target, [1, 2, 3, 4], max_tokens=11)
    spec = generate_speculative(target, draft, [1, 2, 3, 4], max_tokens=11, draft_width=width)
    assert spec.tokens == plain.tokens
    assert spec.state.tokens == plain.state.tokens
    for actual, expected in zip(spec.state.decoder_keys, plain.state.decoder_keys):
        np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=0)


def test_single_pass_draft_input_is_not_consumed_by_validation():
    logits = np.eye(4)[[0, 2, 3]]
    drafts = np.array([1, 2], dtype=object).flat
    result = verify_greedy(logits, drafts)
    assert result.tokens == (0,)
    assert result.accepted == 0


def test_first_rejection_commits_only_target_replacement():
    logits = np.eye(6)[[2, 3, 4, 5]]
    result = verify_greedy(logits, [1, 3, 4])
    assert result.tokens == (2,)
    assert result.accepted == 0


def test_middle_rejection_does_not_commit_the_rejected_suffix():
    logits = np.eye(6)[[1, 2, 3, 4]]
    result = verify_greedy(logits, [1, 5, 3])
    assert result.tokens == (1, 2)
    assert result.accepted == 1


def test_all_accepted_get_a_bonus_token():
    logits = np.eye(6)[[1, 2, 3, 4]]
    result = verify_greedy(logits, [1, 2, 3])
    assert result.tokens == (1, 2, 3, 4)
    assert result.accepted == 3


def test_eos_in_an_accepted_prefix_stops_before_later_drafts():
    logits = np.eye(6)[[1, 2, 3, 4]]
    result = verify_greedy(logits, [1, 2, 3], eos_id=2)
    assert result.tokens == (1, 2)


def test_eos_replacement_stops_generation():
    target = TinyModel(seed=4)
    draft = TinyModel(seed=1, depth=1)
    eos = int(target.prefill([1, 2]).logits[-1].argmax())
    plain = generate(target, [1, 2], max_tokens=8, eos_id=eos)
    spec = generate_speculative(target, draft, [1, 2], max_tokens=8, eos_id=eos)
    assert plain.tokens == spec.tokens == (eos,)
    assert spec.state.tokens == (1, 2, eos)


def test_zero_token_request_keeps_the_prompt_state():
    result = generate_speculative(TinyModel(), TinyModel(depth=1), [1, 2], max_tokens=0)
    assert result.tokens == ()
    assert result.state.tokens == (1, 2)


def test_numpy_integer_draft_width_is_normalized():
    target = TinyModel()
    draft = TinyModel(depth=1)
    actual = generate_speculative(target, draft, [1], max_tokens=3, draft_width=np.int64(1))
    expected = generate_speculative(target, draft, [1], max_tokens=3, draft_width=1)
    assert actual.tokens == expected.tokens


@pytest.mark.parametrize("width", [0, -1, 1.5])
def test_invalid_draft_width_is_rejected(width):
    with pytest.raises(ValueError):
        generate_speculative(TinyModel(), TinyModel(depth=1), [1], draft_width=width)
