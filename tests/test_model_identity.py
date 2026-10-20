import pytest

from core_deepinfer import Engine, TinyModel
from core_deepinfer.cache import StateCache
from core_deepinfer.runtime import Runner


def test_parameter_fingerprint_tracks_the_current_weights_not_just_initialization():
    model = TinyModel(seed=7)
    before = model.identity
    model.head[0, 0] += 1
    assert model.identity != before


def test_an_existing_cursor_cannot_continue_after_its_model_parameters_change():
    model = TinyModel(seed=7)
    runner = Runner(model, model.prefill([1, 2, 3, 4]))
    before = runner.capture()
    model.head.fill(0)
    with pytest.raises(ValueError, match="different model"):
        runner.advance(5)
    assert runner.tokens == before.state.tokens


def test_shared_cache_does_not_reuse_old_logits_after_a_weight_change_between_engines():
    model = TinyModel(seed=7)
    cache = StateCache()
    prompt = [1, 2, 3, 4]
    Engine(model, cache=cache).generate(prompt, max_tokens=0)
    model.head.fill(0)
    expected = int(model.prefill(prompt).logits[-1].argmax())
    result = Engine(model, cache=cache).generate(prompt, max_tokens=1)
    assert result.tokens == (expected,)
    assert result.state.owner == model.identity
