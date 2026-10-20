import pytest

from core_deepinfer import Engine, RequestError, TinyModel


def test_a_reported_failure_does_not_poison_future_requests_on_the_same_engine(monkeypatch):
    model = TinyModel()
    step = model.step
    fail = [True]

    def controlled(state, token):
        if fail[0]:
            raise RuntimeError("one-shot backend failure")
        return step(state, token)

    monkeypatch.setattr(model, "step", controlled)
    engine = Engine(model)
    engine.submit([1, 2, 3], max_tokens=2)
    with pytest.raises(RequestError, match="one-shot"):
        engine.run()
    fail[0] = False
    result = engine.generate([1, 2, 3], max_tokens=2)
    assert result.finish_reason == "length"
    assert result.state.tokens == (*result.prompt, *result.tokens)


def test_a_drain_reports_each_completed_request_once_but_results_remain_addressable():
    engine = Engine()
    first = engine.submit([1, 2], max_tokens=1)
    assert [result.request_id for result in engine.run()] == [first]
    assert engine.run() == []
    second = engine.submit([3, 4], max_tokens=1)
    assert [result.request_id for result in engine.run()] == [second]
    assert engine.result(first).finish_reason == "length"
