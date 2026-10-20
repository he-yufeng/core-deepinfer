from copy import deepcopy

import numpy as np
import pytest

from core_deepinfer.model import TinyModel
from core_deepinfer.runtime import Runner, encoder_state
from core_deepinfer.speculative import generate


def assert_state(actual, expected):
    assert actual.owner == expected.owner
    assert actual.tokens == expected.tokens
    for name in ["encoder_keys", "encoder_values", "source", "shared_keys", "shared_values"]:
        np.testing.assert_allclose(getattr(actual, name), getattr(expected, name), atol=1e-10, rtol=0)
    for name in ["decoder_keys", "decoder_values"]:
        for left, right in zip(getattr(actual, name), getattr(expected, name), strict=True):
            np.testing.assert_allclose(left, right, atol=1e-10, rtol=0)


@pytest.mark.parametrize("seed", [0, 7, 23])
def test_replay_reuses_encoder_and_matches_full_exact_oracle(seed):
    model = TinyModel(seed=seed)
    full = model.prefill([1, 2, 3, 4, 5, 6, 7])
    replayed = model.replay(encoder_state(full.state))
    np.testing.assert_allclose(replayed.logits, full.logits, atol=1e-10, rtol=0)
    assert_state(replayed.state, full.state)
    assert replayed.stats["encoder_rows"] == 0
    assert replayed.stats["consumer_rows"] == 7 * model.depth


@pytest.mark.parametrize("mode", ["mask", "compact"])
def test_replay_bounded_path_preserves_full_source(mode):
    model = TinyModel(seed=2)
    full = model.prefill(list(range(16)))
    replayed = model.replay(encoder_state(full.state), mode=mode, keep=4)
    reference = model.prefill(list(range(16)), mode=mode, keep=4)
    np.testing.assert_allclose(replayed.logits, reference.logits, atol=1e-10, rtol=0)
    assert len(replayed.state.source) == 16
    assert replayed.stats["encoder_rows"] == 0


@pytest.mark.parametrize("seed", [0, 7])
@pytest.mark.parametrize("window", [1, 4, 8])
def test_runner_reads_actual_ring_and_matches_incremental_oracle(seed, window):
    model = TinyModel(seed=seed, window=window)
    prompt = [1, 2, 3, 4, 5]
    runner = Runner(model, model.prefill(prompt))
    for token in [6, 7, 8, 9, 10]:
        runner.advance(token)
        prompt.append(token)
        full = model.prefill(prompt)
        np.testing.assert_allclose(runner.logits, full.logits[-1], atol=1e-10, rtol=0)
        assert_state(runner.state, full.state)
    assert runner.steps == 5
    assert runner.stats["encoder_rows"] == 10


def test_checkpoint_restores_overwritten_real_kv_before_next_forward():
    model = TinyModel(window=2, seed=9)
    runner = Runner(model, model.prefill([1, 2, 3, 4]))
    before = runner.capture()
    checkpoint = runner.checkpoint()
    for token in [12, 13, 14, 15, 16]:
        runner.advance(token)
    assert not np.array_equal(runner.keys.values, checkpoint.keys.values)
    runner.rollback(checkpoint)
    assert_state(runner.state, before.state)
    np.testing.assert_array_equal(runner.logits, before.logits)
    runner.advance(6)
    assert_state(runner.state, model.prefill([1, 2, 3, 4, 6]).state)
    assert runner.steps == 6  # Rollback does not erase work that actually happened.


@pytest.mark.parametrize("seed", [0, 5, 7])
@pytest.mark.parametrize("width", [0, 1, 3, 7])
def test_verification_after_ring_wrap_commits_the_same_greedy_prefix(seed, width):
    model = TinyModel(window=2, seed=seed)
    runner = Runner(model, model.prefill([1, 2, 3, 4]))
    wrong = (int(runner.logits.argmax()) + 1) % model.vocab_size
    decision = runner.verify([wrong] * width, max_tokens=width + 1)
    expected = generate(model, [1, 2, 3, 4], max_tokens=1)
    assert decision.tokens == expected.tokens
    assert decision.accepted == 0
    assert decision.discarded == width
    assert_state(runner.state, expected.state)
    assert runner.steps == width + 1


def test_middle_rejection_restores_the_accepted_checkpoint_not_the_original_prompt():
    model = TinyModel(window=2, seed=3)
    prompt = [1, 2, 3, 4]
    expected = generate(model, prompt, max_tokens=2)
    first, second = expected.tokens
    runner = Runner(model, model.prefill(prompt))
    decision = runner.verify([first, (second + 1) % model.vocab_size, first, second], max_tokens=5)
    assert decision.tokens == expected.tokens
    assert decision.accepted == 1
    assert decision.discarded == 3
    assert_state(runner.state, expected.state)


def test_identical_draft_accepts_all_and_appends_bonus_without_reprefilling(monkeypatch):
    model = TinyModel(seed=4, window=2)
    prompt = [1, 2, 3, 4]
    target = Runner(model, model.prefill(prompt))
    draft = Runner(model, model.prefill(prompt))
    expected = generate(model, prompt, max_tokens=6)
    proposal = draft.propose(5)

    def forbidden(*args, **kwargs):
        raise AssertionError("verification must not prefill the full prefix")

    monkeypatch.setattr(model, "prefill", forbidden)
    decision = target.verify(proposal.tokens, max_tokens=6)
    assert decision.tokens == expected.tokens
    assert decision.accepted == 5
    assert decision.discarded == 0
    assert target.steps == 6
    assert_state(target.state, expected.state)


def test_eos_inside_a_verified_prefix_discards_the_later_physical_writes():
    model = TinyModel(window=1, seed=1)
    prompt = [1, 2, 3]
    baseline = generate(model, prompt, max_tokens=4)
    eos = baseline.tokens[0]
    runner = Runner(model, model.prefill(prompt))
    decision = runner.verify(baseline.tokens[:3], max_tokens=4, eos_id=eos)
    assert decision.tokens == (eos,)
    assert decision.accepted == 1
    assert decision.discarded == 2
    assert_state(runner.state, model.prefill([*prompt, eos]).state)


def test_cached_state_round_trip_is_isolated_from_the_original_runner():
    model = TinyModel()
    first = Runner(model, model.prefill([1, 2, 3]))
    saved = first.capture()
    second = Runner.from_cached(model, saved)
    second.advance(4)
    assert first.tokens == saved.state.tokens == (1, 2, 3)
    saved.state.source[:] = 100
    assert not np.array_equal(first.state.source, saved.state.source)


def test_replay_and_checkpoints_reject_foreign_owners():
    first = TinyModel(seed=1)
    second = TinyModel(seed=2)
    left = Runner(first, first.prefill([1, 2]))
    right = Runner(second, second.prefill([1, 2]))
    with pytest.raises(ValueError, match="another runner"):
        right.rollback(left.checkpoint())
    with pytest.raises(ValueError, match="different model"):
        second.replay(encoder_state(left.state))


def test_verification_failure_restores_the_preproposal_state(monkeypatch):
    model = TinyModel(window=1)
    runner = Runner(model, model.prefill([1, 2, 3]))
    saved = runner.capture()
    step = model.step
    calls = []

    def failing(state, token):
        calls.append(token)
        if len(calls) == 3:
            raise RuntimeError("controlled forward failure")
        return step(state, token)

    monkeypatch.setattr(model, "step", failing)
    with pytest.raises(RuntimeError, match="controlled"):
        runner.verify([4, 5, 6, 7], max_tokens=5)
    assert len(calls) == 3
    assert_state(runner.state, saved.state)
    np.testing.assert_array_equal(runner.logits, saved.logits)


def test_bad_cached_arrays_are_not_accepted_as_a_hot_hit():
    model = TinyModel()
    runner = Runner(model, model.prefill([1, 2]))
    saved = deepcopy(runner.capture())
    saved.state.shared_keys[0, 0] = np.nan
    with pytest.raises(ValueError, match="invalid shape or value"):
        Runner.from_cached(model, saved)
