"""Run deterministic CPU experiments. No weights, keys or network access."""

import argparse
import json
from dataclasses import asdict

import numpy as np

from .budget import choose_budget, reallocate
from .cache import StateCache, StateRing, cache_key
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "experiment", choices=["all", "replay", "budget", "cache", "speculative"], nargs="?", default="all"
    )
    parser.add_argument("--length", type=int, default=32)
    parser.add_argument("--keep", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--json", action="store_true", help="Print the result without the explanatory heading"
    )
    args = parser.parse_args(argv)
    experiments = {
        "replay": lambda: replay_experiment(length=args.length, keep=args.keep, seed=args.seed),
        "budget": budget_experiment,
        "cache": cache_experiment,
        "speculative": lambda: speculative_experiment(seed=args.seed),
    }
    try:
        result = {name: run() for name, run in experiments.items() if args.experiment in (name, "all")}
        if args.experiment in ("all", "replay"):
            result["chunk_layout"] = asdict(chunk_plan(1024, 0, 512, keep=140, window=128, rollback=6))
    except ValueError as error:
        parser.error(str(error))
    if not args.json:
        print("Core DeepInfer: small numerical references, random untrained model, no performance claims.\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0
