import numpy as np
import pytest

from core_deepinfer.cache import StateCache
from core_deepinfer.engine import Engine, EngineConfig, RequestError
from core_deepinfer.model import TinyModel
from core_deepinfer.runtime import Runner
from core_deepinfer.speculative import generate


def assert_matches(actual, expected):
    assert actual.error is None
    assert actual.tokens == expected.tokens
    assert actual.state.tokens == expected.state.tokens
    for name in ["encoder_keys", "encoder_values", "source", "shared_keys", "shared_values"]:
        np.testing.assert_allclose(
            getattr(actual.state, name), getattr(expected.state, name), atol=1e-10, rtol=0
        )
    for name in ["decoder_keys", "decoder_values"]:
        for left, right in zip(getattr(actual.state, name), getattr(expected.state, name), strict=True):
            np.testing.assert_allclose(left, right, atol=1e-10, rtol=0)


@pytest.mark.parametrize("seed", [0, 5, 7])
@pytest.mark.parametrize("window", [1, 4])
@pytest.mark.parametrize("width", [1, 3, 5])
def test_cached_speculative_engine_matches_greedy_and_all_committed_state(seed, window, width):
    target = TinyModel(seed=seed, window=window)
    draft = TinyModel(seed=seed + 1, depth=1, window=window)
    engine = Engine(target, draft, config=EngineConfig(draft_width=width))
    prompt = [1, 2, 3, 4]
    result = engine.generate(prompt, max_tokens=9)
    assert result.finish_reason == "length"
    assert_matches(result, generate(target, prompt, max_tokens=9))
    assert any(event["kind"] == "commit" for event in engine.trace)


def test_target_and_draft_prefill_only_once_not_on_each_verification(monkeypatch):
    target, draft = TinyModel(seed=7), TinyModel(seed=8, depth=1)
    expected = generate(target, [1, 2, 3, 4], max_tokens=15)
    counts = [0, 0]
    for index, model in enumerate([target, draft]):
        original = model.prefill

        def counted(*args, index=index, original=original, **kwargs):
            counts[index] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(model, "prefill", counted)
    engine = Engine(target, draft)
    result = engine.generate([1, 2, 3, 4], max_tokens=15)
    assert_matches(result, expected)
    assert counts == [1, 1]
    assert any(event.get("discarded_target", 0) > target.window for event in engine.trace)


def test_identical_draft_really_accepts_and_commits_bonus_tokens():
    target = TinyModel(seed=3, window=2)
    engine = Engine(target, TinyModel(seed=3, window=2))
    result = engine.generate([1, 2, 3], max_tokens=13)
    assert_matches(result, generate(target, [1, 2, 3], max_tokens=13))
    commits = [event for event in engine.trace if event["kind"] == "commit"]
    assert commits[0]["accepted"] == 5
    assert len(commits[0]["tokens"]) == 6
    assert result.accepted_drafts > 0


def test_real_cache_moves_from_cold_to_hot_to_warm_then_cold():
    now = [0.0]
    target = TinyModel(seed=7)
    engine = Engine(target, cache=StateCache(clock=lambda: now[0]))
    prompt = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    expected = generate(target, prompt, max_tokens=2)
    paths = []
    work = []
    for timestamp in [0.0, 0.0, 3.0, 200.0]:
        now[0] = timestamp
        result = engine.generate(prompt, max_tokens=2)
        assert_matches(result, expected)
        request = engine.requests[result.request_id]
        work.append(dict(request.target.stats))
        paths.append(
            next(
                event["cache"]
                for event in engine.trace
                if event["kind"] == "restore" and event["request_id"] == result.request_id
            )
        )
    assert paths == ["cold", "hot", "warm", "cold"]
    assert [row["encoder_rows"] for row in work] == [11, 2, 2, 11]
    assert work[2]["consumer_rows"] > work[1]["consumer_rows"]


def test_longest_stored_prefix_reuses_state_then_computes_only_new_suffix():
    model = TinyModel(seed=9)
    engine = Engine(model)
    engine.generate([1, 2, 3, 4], max_tokens=0)
    result = engine.generate([1, 2, 3, 4, 5, 6], max_tokens=1)
    assert_matches(result, generate(model, [1, 2, 3, 4, 5, 6], max_tokens=1))
    restore = next(
        event
        for event in engine.trace
        if event["kind"] == "restore" and event["request_id"] == result.request_id
    )
    assert restore["cache"] == "hot"
    assert restore["reused_positions"] == 4
    assert restore["suffix_positions"] == 2
    assert restore["encoder_rows"] == 2


def test_cache_never_reuses_another_models_state():
    cache = StateCache()
    first = Engine(TinyModel(seed=1), cache=cache)
    second = Engine(TinyModel(seed=2), cache=cache)
    first.generate([1, 2, 3, 4], max_tokens=2)
    result = second.generate([1, 2, 3, 4], max_tokens=2)
    assert_matches(result, generate(second.target, [1, 2, 3, 4], max_tokens=2))
    assert next(e for e in second.trace if e["kind"] == "restore")["cache"] == "cold"


def test_bounded_layout_does_not_silently_restore_an_exact_local_state():
    cache = StateCache()
    model = TinyModel(window=2)
    prompt = list(range(16))
    Engine(model, cache=cache).generate(prompt, max_tokens=0)
    bounded = Engine(model, cache=cache, config=EngineConfig(recovery_mode="compact", keep=2))
    result = bounded.generate(prompt, max_tokens=0)
    expected = model.prefill(prompt, mode="compact", keep=2)
    np.testing.assert_allclose(
        result.state.decoder_keys[-1], expected.state.decoder_keys[-1], atol=1e-10, rtol=0
    )
    restore = next(e for e in bounded.trace if e["kind"] == "restore")
    assert restore["cache"] == "warm"
    assert restore["encoder_rows"] == 0
    assert len(result.state.source) == len(prompt)


def test_small_budget_interleaves_more_requests_than_fit_without_starving_them():
    model = TinyModel(seed=4)
    engine = Engine(model, TinyModel(seed=5, depth=1), config=EngineConfig(max_batch_tokens=2, max_active=8))
    ids = [engine.submit([number, 2, 3], max_tokens=3) for number in range(1, 8)]
    results = engine.run()
    assert [result.request_id for result in results] == ids
    for number, result in enumerate(results, 1):
        assert_matches(result, generate(model, [number, 2, 3], max_tokens=3))
    for event in engine.trace:
        if event["kind"] == "budget":
            assert len(event["requests"]) <= 2
            assert sum(event["newer_lengths"]) == event["total_tokens"] <= 2


def test_ragged_lengths_and_graph_padding_never_become_extra_verified_tokens():
    config = EngineConfig(max_batch_tokens=8, cost_curve=((1, 1.0), (2, 1.2), (4, 1.4), (8, 2.5)))
    model = TinyModel(seed=1)
    engine = Engine(model, TinyModel(seed=2, depth=1), config=config)
    for number, limit in enumerate([1, 2, 7, 9], 1):
        engine.submit([number, 2, 3], max_tokens=limit)
    results = engine.run()
    assert [len(result.tokens) for result in results] == [1, 2, 7, 9]
    for tick in range(1, engine.tick + 1):
        events = [event for event in engine.trace if event["tick"] == tick]
        plans = [event for event in events if event["kind"] == "budget"]
        if not plans:
            continue
        plan = plans[0]
        commits = [event for event in events if event["kind"] == "commit"]
        assert sum(event["planned_positions"] for event in commits) == plan["total_tokens"]
        assert sum(event["target_steps"] for event in commits) <= plan["total_tokens"]
        assert plan["total_tokens"] + plan["padding"] == plan["graph_size"] <= config.max_batch_tokens
    for number, (limit, result) in enumerate(zip([1, 2, 7, 9], results, strict=True), 1):
        assert_matches(result, generate(model, [number, 2, 3], max_tokens=limit))


def test_new_confidence_redistribution_changes_executed_widths_not_just_a_printed_plan(monkeypatch):
    target = TinyModel(seed=2)
    draft = TinyModel(seed=2)
    engine = Engine(target, draft, config=EngineConfig(max_batch_tokens=4))
    original = Runner.propose

    def controlled(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        favored = (self.tokens[0] == 1) == (engine.tick == 1)
        result.confidence = (0.99 if favored else 0.1,) * len(result.tokens)
        return result

    monkeypatch.setattr(Runner, "propose", controlled)
    left = engine.submit([1, 2, 3], max_tokens=8)
    right = engine.submit([9, 2, 3], max_tokens=8)
    engine.step()
    engine.step()
    plans = [e for e in engine.trace if e["kind"] == "budget"]
    assert plans[1]["lengths"] == (3, 1)
    assert plans[1]["newer_lengths"] == (1, 3)
    commits = [e for e in engine.trace if e["kind"] == "commit" and e["tick"] == 2]
    assert [(e["request_id"], e["target_steps"]) for e in commits] == [(left, 1), (right, 3)]
    engine.run()


@pytest.mark.parametrize("with_draft", [False, True])
def test_eos_stops_after_the_committed_token_and_leaves_no_proposed_suffix(with_draft):
    model = TinyModel(seed=4, window=1)
    prompt = [1, 2, 3, 4]
    eos = int(model.prefill(prompt).logits[-1].argmax())
    engine = Engine(model, TinyModel(seed=5, depth=1, window=1) if with_draft else None)
    result = engine.generate(prompt, max_tokens=12, eos_id=eos)
    assert result.finish_reason == "stop"
    assert result.tokens == (eos,)
    assert_matches(result, generate(model, prompt, max_tokens=12, eos_id=eos))


def test_queued_and_running_requests_can_be_cancelled_without_dropping_other_results():
    model = TinyModel()
    engine = Engine(model, config=EngineConfig(max_active=1))
    before = engine.submit([1, 2], max_tokens=4)
    running = engine.submit([3, 4], max_tokens=4)
    healthy = engine.submit([5, 6], max_tokens=4)
    assert engine.cancel(before)
    assert engine.result(before).state is None
    engine.step()
    assert engine.cancel(running)
    assert not engine.cancel(running)
    results = engine.run()
    assert [r.finish_reason for r in results] == ["cancelled", "cancelled", "length"]
    assert results[1].state.tokens == (3, 4, *results[1].tokens)
    assert_matches(engine.result(healthy), generate(model, [5, 6], max_tokens=4))


def test_one_forward_failure_is_explicit_after_the_healthy_requests_drain(monkeypatch):
    target = TinyModel(seed=1)
    draft = TinyModel(seed=2, depth=1)
    original = draft.step

    def failing(state, token):
        if state.tokens[0] == 9:
            raise RuntimeError("controlled draft failure")
        return original(state, token)

    monkeypatch.setattr(draft, "step", failing)
    engine = Engine(target, draft)
    bad = engine.submit([9, 2, 3], max_tokens=4)
    good = engine.submit([1, 2, 3], max_tokens=4)
    with pytest.raises(RequestError, match="controlled draft failure") as caught:
        engine.run()
    assert len(caught.value.results) == 2
    assert engine.result(bad).finish_reason == "error"
    assert engine.result(bad).state.tokens == (9, 2, 3)
    assert_matches(engine.result(good), generate(target, [1, 2, 3], max_tokens=4))


@pytest.mark.parametrize(
    "prompt,limit,eos", [([], 1, None), ([True], 1, None), ([1], -1, None), ([1], 1, 99)]
)
def test_invalid_admission_is_explicit_and_does_not_create_a_partial_request(prompt, limit, eos):
    engine = Engine()
    with pytest.raises(ValueError):
        engine.submit(prompt, max_tokens=limit, eos_id=eos)
    assert not engine.requests
    assert not engine.trace


def test_zero_length_request_returns_the_prompt_state_and_no_tokens():
    model = TinyModel()
    engine = Engine(model, TinyModel(depth=1))
    result = engine.generate([1, 2, 3], max_tokens=0)
    assert_matches(result, generate(model, [1, 2, 3], max_tokens=0))
    assert result.finish_reason == "length"
    assert not [e for e in engine.trace if e["kind"] == "budget"]


def test_pending_result_and_duplicate_id_are_explicit_errors():
    engine = Engine()
    key = engine.submit([1, 2], request_id="chosen")
    with pytest.raises(ValueError, match="not finished"):
        engine.result(key)
    with pytest.raises(ValueError, match="unique"):
        engine.submit([1, 2], request_id="chosen")
    engine.run()
    assert engine.result(key).finish_reason == "length"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_active": 0},
        {"max_batch_tokens": -1},
        {"state_ttl": 0},
        {"prefix_ttl": float("nan")},
        {"recovery_mode": "bad"},
        {"max_batch_tokens": 1, "cost_curve": ((2, 1.0),)},
    ],
)
def test_invalid_engine_configuration_is_rejected_before_execution(kwargs):
    with pytest.raises(ValueError):
        EngineConfig(**kwargs)
