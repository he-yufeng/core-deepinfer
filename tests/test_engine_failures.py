import numpy as np
import pytest

from core_deepinfer.cache import StateCache
from core_deepinfer.engine import Engine, RequestError
from core_deepinfer.model import TinyModel
from core_deepinfer.runtime import Runner


def test_auto_generated_request_ids_skip_names_already_chosen_by_the_caller():
    engine = Engine()
    chosen = engine.submit([1, 2], request_id="request-2", max_tokens=1)
    automatic = engine.submit([3, 4], max_tokens=1)
    assert automatic != chosen
    assert len(engine.run()) == 2


def test_invalid_model_kv_is_rejected_before_either_ring_is_written(monkeypatch):
    model = TinyModel(window=1)
    runner = Runner(model, model.prefill([1, 2, 3]))
    before = runner.capture()
    step = model.step

    def malformed(state, token):
        forward = step(state, token)
        forward.state.decoder_values = tuple(layer[:, :1] for layer in forward.state.decoder_values)
        return forward

    monkeypatch.setattr(model, "step", malformed)
    with pytest.raises(ValueError):
        runner.advance(4)
    assert runner.keys.next_position == runner.values.next_position == 3
    assert runner.state.tokens == before.state.tokens
    np.testing.assert_array_equal(runner.state.decoder_keys[0], before.state.decoder_keys[0])
    np.testing.assert_array_equal(runner.state.decoder_values[0], before.state.decoder_values[0])


def test_cache_write_failure_keeps_committed_tokens_and_state_consistent(monkeypatch):
    cache = StateCache()
    put = cache.put_state

    def failing(key, value, *, ttl):
        if value.state.tokens[0] == 9 and len(value.state.tokens) > 3:
            raise OSError("controlled cache write failure")
        return put(key, value, ttl=ttl)

    monkeypatch.setattr(cache, "put_state", failing)
    engine = Engine(TinyModel(), cache=cache)
    bad = engine.submit([9, 2, 3], max_tokens=1)
    good = engine.submit([1, 2, 3], max_tokens=1)
    with pytest.raises(RequestError, match="controlled cache write failure"):
        engine.run()
    failed = engine.result(bad)
    assert failed.finish_reason == "error"
    assert failed.state.tokens == (*failed.prompt, *failed.tokens)
    assert engine.result(good).finish_reason == "length"
