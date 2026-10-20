import numpy as np
import pytest

from core_deepinfer.model import TinyModel
from core_deepinfer.replay import chunk_plan, replay_experiment


@pytest.mark.parametrize("seed", [0, 7, 21])
@pytest.mark.parametrize("length,keep", [(8, 4), (17, 5), (32, 8)])
def test_mask_and_compact_follow_the_same_bounded_graph(seed, length, keep):
    model = TinyModel(seed=seed, window=4)
    tokens = np.arange(length) % model.vocab_size
    mask = model.prefill(tokens, mode="mask", keep=keep)
    compact = model.prefill(tokens, mode="compact", keep=keep)
    np.testing.assert_allclose(mask.logits[-keep:], compact.logits, atol=1e-10, rtol=0)
    assert compact.stats["consumer_rows"] == keep * model.depth
    assert mask.stats["consumer_rows"] == length * model.depth
    assert mask.stats["encoder_rows"] == compact.stats["encoder_rows"]
    for actual, expected in zip(compact.state.decoder_keys, mask.state.decoder_keys):
        np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=0)


@pytest.mark.parametrize("seed", [0, 3, 11])
def test_incremental_forward_matches_full_exact_oracle(seed):
    model = TinyModel(seed=seed, window=4)
    tokens = [1, 4, 9, 3, 2, 8, 7, 6]
    state = model.prefill(tokens[:2]).state
    for index, token in enumerate(tokens[2:], 2):
        step = model.step(state, token)
        full = model.prefill(tokens[: index + 1])
        np.testing.assert_allclose(step.logits[-1], full.logits[-1], atol=1e-10, rtol=0)
        for actual, expected in zip(step.state.decoder_keys, full.state.decoder_keys):
            np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=0)
        state = step.state


def test_future_tokens_cannot_change_earlier_outputs():
    model = TinyModel(seed=4)
    first = model.prefill([1, 2, 3, 4, 5]).logits
    second = model.prefill([1, 2, 3, 9, 10]).logits
    np.testing.assert_array_equal(first[:3], second[:3])


def test_bounded_replay_is_not_claimed_to_be_exact():
    result = replay_experiment(length=24, keep=4, seed=7)
    assert result["mask_compact_max_error"] < 1e-10
    assert result["bounded_exact_max_error"] > 1e-8
    assert result["model_weights"] == "random, untrained"


def test_compact_does_not_cut_shared_encoder_source():
    model = TinyModel(window=4)
    result = model.prefill(list(range(16)), mode="compact", keep=4)
    assert result.logits.shape == (4, model.vocab_size)
    assert len(result.state.source) == 16
    assert result.state.tokens == tuple(range(16))


def test_numpy_layout_scalars_do_not_overflow_in_python_index_arithmetic():
    expected = chunk_plan(8, 5, 3, keep=4, window=4)
    actual = chunk_plan(np.uint64(8), 5, np.int8(3), keep=np.uint8(4), window=np.int8(4))
    assert actual == expected
    assert actual.positions == (5, 6, 7)


def test_early_chunk_keeps_a_placeholder_without_a_ring_write():
    plan = chunk_plan(1024, 0, 512, keep=140, window=128, rollback=6)
    assert plan.positions == (511,)
    assert plan.ring_slots == (-1,)
    final = chunk_plan(1024, 512, 512, keep=140, window=128, rollback=6)
    assert final.positions == tuple(range(884, 1024))
    assert final.ring_slots[:6] == (-1,) * 6
    assert all(slot >= 0 for slot in final.ring_slots[6:])


@pytest.mark.parametrize("mode,keep", [("bad", None), ("mask", 0), ("compact", 3)])
def test_invalid_replay_configuration_is_rejected(mode, keep):
    with pytest.raises(ValueError):
        TinyModel(window=4).prefill([1, 2, 3, 4, 5], mode=mode, keep=keep)


@pytest.mark.parametrize("tokens", [[], [-1], [1000], [True], [1.5]])
def test_invalid_tokens_are_rejected(tokens):
    with pytest.raises(ValueError):
        TinyModel().prefill(tokens)
