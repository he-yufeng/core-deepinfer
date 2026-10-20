import inspect
import sys

import numpy as np
import pytest

from core_deepinfer import Engine, TinyModel
from core_deepinfer.cache import StateCache, cache_key
from core_deepinfer.demo import state_error
from core_deepinfer.runtime import Runner


def interrupt_before(function, marker, action):
    lines, start = inspect.getsourcelines(function)
    line = next(start + index for index, text in enumerate(lines) if marker in text)
    previous = sys.gettrace()
    fired = False

    def trace(frame, event, arg):
        nonlocal fired
        if frame.f_code is not function.__code__:
            return None
        if event == "line" and frame.f_lineno == line and not fired:
            fired = True
            sys.settrace(None)
            raise KeyboardInterrupt()
        return trace

    sys.settrace(trace)
    try:
        with pytest.raises(KeyboardInterrupt):
            action()
    finally:
        sys.settrace(previous)
    assert fired


@pytest.mark.parametrize("kind", ["prefix", "state"])
def test_interrupted_cache_insert_stays_within_the_configured_capacity(kind):
    cache = StateCache(capacity=1)
    put = cache.put_prefix if kind == "prefix" else cache.put_state
    first = cache_key("model", "layout", [1])
    second = cache_key("model", "layout", [2])
    put(first, np.array([1.0]), ttl=60)
    interrupt_before(StateCache._put, "while len(table)", lambda: put(second, np.array([2.0]), ttl=60))
    table = cache.prefixes if kind == "prefix" else cache.states
    assert len(table) <= cache.capacity
    for key, (_, value) in table.items():
        np.testing.assert_array_equal(value, [1.0] if key == first else [2.0])


def test_interruption_after_result_publication_leaves_a_consistent_terminal_status():
    engine = Engine()
    key = engine.submit([1, 2, 3], max_tokens=1)
    interrupt_before(Engine._finish, 'request.status = "finished"', engine.step)
    assert key in engine._results
    assert engine.requests[key].status == "finished"
    assert engine.result(key).finish_reason == "length"
    assert len(engine.run()) == 1


def test_system_exit_is_propagated_after_runner_cleanup(monkeypatch):
    model = TinyModel(window=1)
    runner = Runner(model, model.prefill([1, 2, 3]))
    before = runner.capture()
    step = model.step
    calls = [0]

    def exiting(state, token):
        calls[0] += 1
        if calls[0] == 2:
            raise SystemExit(42)
        return step(state, token)

    with monkeypatch.context() as context:
        context.setattr(model, "step", exiting)
        with pytest.raises(SystemExit) as caught:
            runner.propose(4)
    assert caught.value.code == 42
    assert runner.tokens == before.state.tokens
    assert state_error(runner.state, before.state) < 1e-10
