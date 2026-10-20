import numpy as np
import pytest

from core_deepinfer import Engine, EngineConfig, TinyModel
from core_deepinfer.cache import StateCache
from core_deepinfer.demo import state_error
from core_deepinfer.speculative import generate


@pytest.mark.parametrize("seed", range(12))
def test_mixed_request_lifecycles_match_a_separate_greedy_oracle(seed):
    rng = np.random.default_rng(seed)
    window = [1, 2, 4, 8][seed % 4]
    model = TinyModel(seed=seed, width=8, vocab_size=16, depth=2, window=window)
    draft = (
        None if seed % 3 == 0 else TinyModel(seed=seed + 1, width=8, vocab_size=16, depth=1, window=window)
    )
    cap = [1, 2, 4, 7][seed % 4]
    curve = tuple((size, 1 + size * 0.2) for size in sorted({1, min(2, cap), cap})) if seed % 2 else None
    config = EngineConfig(max_batch_tokens=cap, max_active=4, draft_width=7, cost_curve=curve)
    clock = [0.0]
    cache = StateCache(capacity=[1, 2, 16][seed % 3], clock=lambda: clock[0])
    engine = Engine(model, draft, config=config, cache=cache)
    for timestamp in [0.0, 3.0, 200.0]:
        clock[0] = timestamp
        ids = []
        for index in range(4):
            prompt = [1, 2, *map(int, rng.integers(0, 16, size=index + 1))]
            limit = int(rng.integers(0, 10))
            eos = int(model.prefill(prompt).logits[-1].argmax()) if index == 1 else None
            ids.append(engine.submit(prompt, max_tokens=limit, eos_id=eos))
        engine.cancel(ids[-1])
        engine.step()
        results = engine.run()
        assert [result.request_id for result in results] == ids
        for result in results:
            request = engine.requests[result.request_id]
            if result.finish_reason == "cancelled":
                assert result.tokens == () and result.state is None
                continue
            expected = generate(model, request.prompt, max_tokens=request.max_tokens, eos_id=request.eos_id)
            assert result.tokens == expected.tokens
            assert result.state.tokens == expected.state.tokens
            assert state_error(result.state, expected.state) < 1e-10
        assert engine.run() == []
    for event in engine.trace:
        if event["kind"] == "budget":
            assert sum(event["newer_lengths"]) == event["total_tokens"] <= cap
        elif event["kind"] == "commit":
            assert event["target_steps"] <= event["planned_positions"]
