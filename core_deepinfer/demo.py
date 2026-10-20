"""Run the real engine through cache hits, expiry, verification and cancellation."""

import numpy as np

from .cache import StateCache
from .engine import Engine, EngineConfig
from .model import TinyModel
from .speculative import generate


def state_error(actual, expected):
    errors = [0.0]
    for name in ["encoder_keys", "encoder_values", "source", "shared_keys", "shared_values"]:
        errors.append(float(np.max(np.abs(getattr(actual, name) - getattr(expected, name)))))
    for name in ["decoder_keys", "decoder_values"]:
        errors.extend(
            float(np.max(np.abs(left - right)))
            for left, right in zip(getattr(actual, name), getattr(expected, name), strict=True)
        )
    return max(errors)


def run_demo(*, seed=7, include_trace=False):
    now = [0.0]
    target = TinyModel(seed=seed)
    draft = TinyModel(seed=seed + 1, depth=1)
    config = EngineConfig(
        max_batch_tokens=8,
        cost_curve=((1, 1.0), (2, 1.2), (4, 1.4), (8, 2.5)),
    )
    engine = Engine(target, draft, config=config, cache=StateCache(clock=lambda: now[0]))
    stages = []
    same_tokens = True
    max_error = 0.0

    for label, timestamp in [
        ("cold_and_prefix_reuse", 0.0),
        ("hot", 0.0),
        ("warm", 3.0),
        ("cold_again", 200.0),
    ]:
        now[0] = timestamp
        start = len(engine.trace)
        before = engine.metrics()
        ids = [engine.submit([1, 2, 3, 4], max_tokens=8 if not stages else 4)]
        if not stages:
            ids.append(engine.submit([1, 2, 3, 4, 5], max_tokens=5))
            cancelled = engine.submit([9, 8, 7], max_tokens=4)
            engine.cancel(cancelled)
            ids.append(cancelled)
        engine.run()
        outputs = [engine.result(key) for key in ids]
        after = engine.metrics()
        for output in outputs:
            if output.finish_reason == "cancelled":
                continue
            expected = generate(target, output.prompt, max_tokens=len(output.tokens))
            same_tokens &= output.tokens == expected.tokens
            max_error = max(max_error, state_error(output.state, expected.state))
        events = engine.trace[start:]
        stage = {
            "name": label,
            "fake_clock_seconds": timestamp,
            "results": [output.to_dict() for output in outputs],
            "restores": [event for event in events if event["kind"] == "restore"],
            "work": {
                role: {key: after[role][key] - before[role][key] for key in after[role]} for role in after
            },
            "target_rolled_back_positions": sum(event.get("discarded_target", 0) for event in events),
            "budget_rounds": sum(event["kind"] == "budget" for event in events),
        }
        if include_trace:
            stage["events"] = events
        stages.append(stage)
    try:
        engine.submit([target.vocab_size], max_tokens=1)
    except ValueError as error:
        rejected = str(error)
    else:
        raise AssertionError("the invalid prompt should not have entered the queue")
    if not same_tokens or max_error > 1e-10:
        raise AssertionError("engine output or committed state differs from the greedy oracle")
    return {
        "scope": "Real NumPy CPU engine; random untrained weights; no language or throughput claim",
        "clock_scope": "Injected fake clock; no waiting or measured latency",
        "budget_scope": "Synthetic costs and draft self-probabilities, not DSpark's learned confidence",
        "same_tokens_as_greedy": bool(same_tokens),
        "committed_state_max_error": max_error,
        "rejected_prompt_error": rejected,
        "stages": stages,
        "work": engine.metrics(),
        "work_scope": "Engine query rows only; excludes oracle checks; not a FLOP or speed estimate",
    }
