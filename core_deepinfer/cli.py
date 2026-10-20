"""Run deterministic CPU experiments. No weights, keys or network access."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .budget import choose_budget, reallocate
from .cache import StateCache, StateRing, cache_key
from .demo import run_demo
from .engine import Engine, EngineConfig
from .model import TinyModel
from .replay import chunk_plan, replay_experiment
from .speculative import generate, generate_speculative


def budget_experiment():
    stale = [[0.95, 0.9, 0.8, 0.7, 0.5], [0.65, 0.5, 0.4, 0.3, 0.2]]
    newer = list(reversed(stale))
    curve = [(2, 1.0), (4, 1.3), (8, 2.8), (12, 6.0)]
    budget = choose_budget(stale, curve, sampler_ms=0.4)
    return {
        "curve_source": "synthetic teaching input, not a hardware measurement",
        "curve": curve,
        "cpu_budget": asdict(budget),
        "newer_confidence_lengths": reallocate(budget.total_tokens, newer),
        "padding_positions": budget.graph_size - budget.total_tokens,
        "scope": "Reference allocation only; not GPU execution or an asynchronous engine",
    }


def cache_experiment():
    now = [0.0]
    cache = StateCache(clock=lambda: now[0])
    key = cache_key("random-model", "float64-window4", [1, 2, 3, 4], alignment=4)
    cache.put_prefix(key, b"shared encoder state", ttl=100)
    cache.put_state(key, b"local window state", ttl=3)
    before = cache.get_state(key) is not None
    now[0] = 3
    state_expired = cache.get_state(key) is None
    prefix_survives = cache.get_prefix(key) is not None
    ring = StateRing(3, 1)
    for number in range(3):
        ring.append([number])
    snapshot = ring.checkpoint()
    for number in range(3, 6):
        ring.append([number])
    ring.rollback(snapshot)
    return {
        "state_before_expiry": before,
        "state_expired_at_ttl": state_expired,
        "long_prefix_survives": prefix_survives,
        "ring_after_overwrite_and_rollback": [ring.read(i).tolist() for i in range(3)],
        "scope": "Fake clock and in-process copied arrays, not a distributed storage benchmark",
    }


def speculative_experiment(*, seed=7):
    target = TinyModel(seed=seed)
    draft = TinyModel(seed=seed + 1, depth=1)
    prompt = [1, 2, 3, 4]
    ordinary = generate(target, prompt, max_tokens=16)
    speculative = generate_speculative(target, draft, prompt, max_tokens=16, draft_width=5)
    error = max(
        float(np.max(np.abs(a - b)))
        for a, b in zip(ordinary.state.decoder_keys, speculative.state.decoder_keys)
    )
    return {
        "model_weights": "random, untrained",
        "greedy_tokens": ordinary.tokens,
        "speculative_tokens": speculative.tokens,
        "same_tokens": ordinary.tokens == speculative.tokens,
        "committed_state_max_error": error,
        "accepted_drafts": speculative.accepted_drafts,
        "proposed_drafts": speculative.draft_tokens,
        "scope": "Recomputing greedy reference, not DSpark or a throughput optimization",
    }


def generate_requests(args):
    model = TinyModel(seed=args.seed, window=args.window)
    draft = None if args.greedy else TinyModel(seed=args.seed + 1, depth=1, window=args.window)
    config = EngineConfig(
        max_batch_tokens=args.batch_budget,
        draft_width=args.draft_width,
        recovery_mode=args.recovery,
        keep=args.keep,
    )
    engine = Engine(model, draft, config=config)
    if args.input is not None:
        jobs = json.loads(Path(args.input).read_text(encoding="utf-8"))
        if not isinstance(jobs, list) or not jobs:
            raise ValueError("input must be a nonempty JSON array of request objects")
    else:
        jobs = []
        for prompt in args.prompt or ["1,2,3,4"]:
            parts = prompt.split(",")
            if not all(part.strip() for part in parts):
                raise ValueError("prompt must contain comma-separated token IDs")
            jobs.append({"prompt": [int(part) for part in parts]})
    for job in jobs:
        if (
            not isinstance(job, dict)
            or "prompt" not in job
            or set(job) - {"prompt", "max_tokens", "eos_id", "request_id"}
        ):
            raise ValueError("each request needs prompt and optional max_tokens, eos_id, request_id")
        engine.submit(
            job["prompt"],
            max_tokens=job.get("max_tokens", args.max_tokens),
            eos_id=job.get("eos_id", args.eos_id),
            request_id=job.get("request_id"),
        )
    results = engine.run()
    output = {
        "scope": "Real CPU execution with random untrained weights; token IDs are not language output",
        "recovery_mode": args.recovery,
        "exact_recovery": args.recovery == "exact",
        "results": [result.to_dict() for result in results],
        "work": engine.metrics(),
        "work_scope": "Query-row counts, not FLOPs or an inference speedup",
    }
    if args.trace:
        output["events"] = engine.trace
    return output


def print_engine(output):
    print("Core DeepInfer | NumPy CPU | random, untrained weights")
    if "stages" in output:
        print(f"Greedy output matches: {output['same_tokens_as_greedy']}")
        print(f"Committed state max error: {output['committed_state_max_error']:.3g}")
        for stage in output["stages"]:
            restores = ", ".join(
                f"{event['request_id']}={event['cache']}({event['reused_positions']})"
                for event in stage["restores"]
                if event["role"] == "target"
            )
            print(f"\n{stage['name']} | fake clock {stage['fake_clock_seconds']:g}s | {restores}")
            for result in stage["results"]:
                print(f"  {result['request_id']}: {result['finish_reason']}, tokens={list(result['tokens'])}")
            print(f"  target rollback: {stage['target_rolled_back_positions']} positions")
    else:
        for result in output["results"]:
            print(f"{result['request_id']}: {result['finish_reason']}, tokens={list(result['tokens'])}")
    print(f"\nActual target query rows: {output['work']['target']}")
    print("Row counts are not a speedup. Demo costs are synthetic; weights are not trained.")
    events = output.get("events", [])
    for stage in output.get("stages", []):
        events.extend(stage.get("events", []))
    if events:
        print("\nExecution trace:")
        for event in events:
            print(json.dumps(event, ensure_ascii=False))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "experiment",
        choices=["demo", "generate", "all", "replay", "budget", "cache", "speculative"],
        nargs="?",
        default="demo",
    )
    parser.add_argument("--length", type=int, default=32, help="Replay experiment input length")
    parser.add_argument("--keep", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--prompt", action="append", help="Comma-separated token IDs; repeat for a batch")
    inputs.add_argument("--input", help="JSON request file for generate")
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--eos-id", type=int)
    parser.add_argument("--draft-width", type=int, default=5)
    parser.add_argument("--batch-budget", type=int, default=16)
    parser.add_argument("--window", type=int, default=4)
    parser.add_argument("--recovery", choices=["exact", "mask", "compact"], default="exact")
    parser.add_argument("--greedy", action="store_true", help="Disable the draft model in generate")
    parser.add_argument("--trace", action="store_true", help="Include actual engine execution events")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable results")
    args = parser.parse_args(argv)
    experiments = {
        "replay": lambda: replay_experiment(length=args.length, keep=args.keep, seed=args.seed),
        "budget": budget_experiment,
        "cache": cache_experiment,
        "speculative": lambda: speculative_experiment(seed=args.seed),
    }
    try:
        if args.experiment == "demo":
            result = {"demo": run_demo(seed=args.seed, include_trace=args.trace)}
        elif args.experiment == "generate":
            result = {"generation": generate_requests(args)}
        else:
            result = {name: run() for name, run in experiments.items() if args.experiment in (name, "all")}
            if args.experiment in ("all", "replay"):
                result["chunk_layout"] = asdict(chunk_plan(1024, 0, 512, keep=140, window=128, rollback=6))
    except (ValueError, OSError) as error:
        parser.error(str(error))
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    elif args.experiment in ("demo", "generate"):
        print_engine(next(iter(result.values())))
    else:
        print("Core DeepInfer: numerical detail experiments, not hardware benchmarks.\n")
        print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0
