from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from core_deepinfer import Engine, EngineConfig, TinyModel
from core_deepinfer.cache import StateCache
from core_deepinfer.demo import state_error
from core_deepinfer.runtime import Runner
from core_deepinfer.speculative import generate


def assert_committed(request):
    expected = (*request.prompt, *request.tokens)
    for runner in [request.target, request.draft]:
        if runner is not None:
            assert runner.tokens == expected
            oracle = runner.model.prefill(expected)
            assert state_error(runner.state, oracle.state) < 1e-10


def assert_resumes(engine, ids):
    queued = list(engine._queue)
    assert set(queued) == set(ids)
    assert len(queued) == len(set(queued))
    results = engine.run()
    assert [result.request_id for result in results] == ids
    for result in results:
        request = engine.requests[result.request_id]
        expected = generate(
            engine.target, request.prompt, max_tokens=request.max_tokens, eos_id=request.eos_id
        )
        assert result.tokens == expected.tokens
        assert state_error(result.state, expected.state) < 1e-10
        assert_committed(request)


@pytest.mark.parametrize("operation", ["propose", "verify"])
def test_runner_interrupt_restores_real_kv_and_source_before_propagating(operation, monkeypatch):
    model = TinyModel(window=2)
    runner = Runner(model, model.prefill([1, 2, 3, 4]))
    before = runner.capture()
    step = model.step
    calls = []

    def interrupted(state, token):
        calls.append(token)
        if len(calls) == 3:
            raise KeyboardInterrupt()
        return step(state, token)

    with monkeypatch.context() as context:
        context.setattr(model, "step", interrupted)
        with pytest.raises(KeyboardInterrupt):
            if operation == "propose":
                runner.propose(5)
            else:
                runner.verify([5, 6, 7, 8, 9], max_tokens=6)
    assert runner.tokens == before.state.tokens
    assert state_error(runner.state, before.state) < 1e-10
    np.testing.assert_array_equal(runner.logits, before.logits)
    assert runner.steps == 2
    runner.advance(5)
    assert state_error(runner.state, model.prefill([1, 2, 3, 4, 5]).state) < 1e-10


@pytest.mark.parametrize("interrupt_at", [3, 7])
def test_interrupt_during_draft_requeues_earlier_proposals_and_the_current_request(interrupt_at, monkeypatch):
    target = TinyModel(seed=7, window=2)
    draft = TinyModel(seed=8, depth=1, window=2)
    engine = Engine(target, draft, config=EngineConfig(max_active=3, max_batch_tokens=18))
    ids = [engine.submit([number, 2, 3, 4], max_tokens=8) for number in [1, 5, 9]]
    step = draft.step
    calls = [0]

    def interrupted(state, token):
        calls[0] += 1
        if calls[0] == interrupt_at:
            raise KeyboardInterrupt()
        return step(state, token)

    with monkeypatch.context() as context:
        context.setattr(draft, "step", interrupted)
        with pytest.raises(KeyboardInterrupt):
            engine.step()
    for key in ids:
        assert_committed(engine.requests[key])
    assert_resumes(engine, ids)


@pytest.mark.parametrize("interrupt_at", [3, 8])
def test_interrupt_during_verification_preserves_completed_commits_only(interrupt_at, monkeypatch):
    target = TinyModel(seed=7, window=2)
    draft = TinyModel(seed=7, window=2)
    engine = Engine(target, draft, config=EngineConfig(max_active=3, max_batch_tokens=18))
    ids = [engine.submit([number, 2, 3, 4], max_tokens=8) for number in [1, 5, 9]]
    step = target.step
    calls = [0]

    def interrupted(state, token):
        calls[0] += 1
        if calls[0] == interrupt_at:
            raise KeyboardInterrupt()
        return step(state, token)

    with monkeypatch.context() as context:
        context.setattr(target, "step", interrupted)
        with pytest.raises(KeyboardInterrupt):
            engine.step()
    assert len(engine.requests[ids[0]].tokens) == (6 if interrupt_at == 8 else 0)
    for key in ids:
        assert_committed(engine.requests[key])
    assert_resumes(engine, ids)


def test_interrupt_between_target_verification_and_draft_sync_is_atomic(monkeypatch):
    target, draft = TinyModel(seed=3, window=2), TinyModel(seed=3, window=2)
    engine = Engine(target, draft, config=EngineConfig(max_active=2, max_batch_tokens=12))
    ids = [engine.submit([number, 2, 3, 4], max_tokens=8) for number in [1, 5]]
    step = draft.step
    calls = [0]

    def interrupted(state, token):
        calls[0] += 1
        if calls[0] == 11:  # Both proposals exist; the first target has verified and is syncing its bonus.
            raise KeyboardInterrupt()
        return step(state, token)

    with monkeypatch.context() as context:
        context.setattr(draft, "step", interrupted)
        with pytest.raises(KeyboardInterrupt):
            engine.step()
    for key in ids:
        assert not engine.requests[key].tokens
        assert_committed(engine.requests[key])
    assert_resumes(engine, ids)


@pytest.mark.parametrize("reason", ["length", "stop"])
def test_interrupt_while_persisting_terminal_state_does_not_lose_the_result_or_repeat_eos(
    reason, monkeypatch
):
    model = TinyModel(seed=7)
    cache = StateCache()
    engine = Engine(model, cache=cache)
    prompt = [1, 2, 3, 4]
    eos = int(model.prefill(prompt).logits[-1].argmax()) if reason == "stop" else None
    key = engine.submit(prompt, max_tokens=8 if eos is not None else 1, eos_id=eos)
    put = cache.put_state

    def interrupted(cache_key, value, *, ttl):
        if len(value.state.tokens) > len(prompt):
            raise KeyboardInterrupt()
        return put(cache_key, value, ttl=ttl)

    with monkeypatch.context() as context:
        context.setattr(cache, "put_state", interrupted)
        with pytest.raises(KeyboardInterrupt):
            engine.step()
    request = engine.requests[key]
    assert len(request.tokens) == 1
    assert_committed(request)
    assert key in engine._queue or key in engine._results
    results = engine.run()
    assert len(results) == 1
    assert results[0].finish_reason == reason
    assert len(results[0].tokens) == 1
    assert request.target.steps == 1


def test_interrupt_in_draft_initialization_can_resume_without_silently_disabling_speculation(monkeypatch):
    target, draft = TinyModel(), TinyModel(seed=1, depth=1)
    engine = Engine(target, draft)
    key = engine.submit([1, 2, 3, 4], max_tokens=8)

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt()

    with monkeypatch.context() as context:
        context.setattr(draft, "prefill", interrupted)
        with pytest.raises(KeyboardInterrupt):
            engine.step()
    assert_resumes(engine, [key])
    assert engine.requests[key].draft is not None
    assert engine.result(key).proposed_drafts > 0


def test_batch_planner_failure_restores_all_prefetched_draft_branches(monkeypatch):
    import core_deepinfer.engine as module

    engine = Engine(TinyModel(), TinyModel(seed=1, depth=1))
    ids = [engine.submit([number, 2, 3], max_tokens=8) for number in [1, 5]]

    def interrupted(*args, **kwargs):
        raise ValueError("controlled planner failure")

    with monkeypatch.context() as context:
        context.setattr(module, "choose_budget", interrupted)
        with pytest.raises(ValueError, match="controlled planner"):
            engine.step()
    for key in ids:
        assert_committed(engine.requests[key])
    assert_resumes(engine, ids)


def test_validated_cost_curve_does_not_alias_mutable_input_rows():
    curve = [[1, 1.0], [2, 1.2], [4, 2.0]]
    config = EngineConfig(cost_curve=curve)
    curve[0][1] = 0
    assert config.cost_curve == ((1, 1.0), (2, 1.2), (4, 2.0))


def test_mutating_an_originally_empty_curve_cannot_make_the_queue_stop_progressing():
    curve = []
    config = EngineConfig(cost_curve=curve)
    engine = Engine(config=config)
    key = engine.submit([1, 2], max_tokens=1)
    curve.append((0, 0.0))
    engine.step()
    assert key in engine._results


def test_validated_configuration_cannot_be_changed_to_zero_active_requests():
    config = EngineConfig()
    with pytest.raises(FrozenInstanceError):
        config.max_active = 0
