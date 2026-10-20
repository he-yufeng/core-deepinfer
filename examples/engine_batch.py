"""Run with: python examples/engine_batch.py (after installing the package)."""

from core_deepinfer import Engine, EngineConfig, TinyModel
from core_deepinfer.demo import state_error
from core_deepinfer.speculative import generate


def main():
    target = TinyModel(seed=7)
    engine = Engine(target, TinyModel(seed=8, depth=1), config=EngineConfig(max_batch_tokens=4))
    prompts = [[1, 2, 3, 4], [1, 2, 3, 4, 5], [9, 8, 7]]
    ids = [engine.submit(prompt, max_tokens=6) for prompt in prompts]
    cancelled = engine.submit([2, 4, 6], max_tokens=4)
    engine.cancel(cancelled)
    events = engine.step()
    assert any(event["kind"] == "commit" for event in events)
    results = engine.run()
    assert len(results) == 4
    for request_id, prompt in zip(ids, prompts, strict=True):
        result = engine.result(request_id)
        expected = generate(target, prompt, max_tokens=6)
        assert result.tokens == expected.tokens
        assert state_error(result.state, expected.state) < 1e-10
        print(request_id, result.tokens, result.finish_reason)
    assert engine.result(cancelled).finish_reason == "cancelled"
    assert engine.run() == []
    again = engine.generate(prompts[0], max_tokens=3)
    assert len(again.tokens) == 3
    print("cache reused; next request:", again.request_id, again.tokens)
    print("actual query rows, not a speed estimate:", engine.metrics())


if __name__ == "__main__":
    main()
