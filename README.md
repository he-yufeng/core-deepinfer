# Core DeepInfer

**Read the mechanism. Run the counterexample. Change it and test again.**

[中文](README_CN.md) · [Source map](docs/SOURCE_MAP.md) · [Validation](docs/VALIDATION.md)

[![CPU reference checks](https://github.com/he-yufeng/core-deepinfer/actions/workflows/tests.yml/badge.svg)](https://github.com/he-yufeng/core-deepinfer/actions/workflows/tests.yml)

A small, executable companion to DeepInfer's state and scheduling ideas. Instead of wrapping another chat model, it makes four questions inspectable on a CPU:

- Does a smaller attention mask actually remove query computation?
- Why can bounded replay match compact replay but differ from exact recovery?
- What happens when speculative work overwrites a window and is then rejected?
- How can newer confidence redistribute work without changing the total budget?

The model performs real float64 attention, projections and feed-forward operations with **random, untrained parameters**. It has no language capability and does not load DeepSeek weights. The budget experiment uses **synthetic costs**, not measured GPU timings. This is an independent teaching project, not a smaller production DeepInfer or a benchmark reproduction.

## Run the experiments

Python 3.10+ and NumPy. No GPU, model download, service or API key.

```bash
git clone https://github.com/he-yufeng/core-deepinfer.git
cd core-deepinfer
python -m pip install -e '.[dev]'
python -m core_deepinfer all
python -m pytest -q
```

You can also run from the checkout with NumPy already installed:

```bash
python -m core_deepinfer replay --length 32 --keep 8 --seed 7 --json
python -m core_deepinfer speculative --json
python -m core_deepinfer cache --json
python -m core_deepinfer budget --json
```

`all` runs every experiment. JSON output carries its evidence scope alongside the numbers.

## Start with a result you can explain

For the default replay case, three consumer layers process a 32-token input. Both paths keep the same shared encoder source. `mask` processes all query rows, while `compact` processes the last eight:

```text
mask consumer rows:       96
compact consumer rows:    24
shared source rows:       32
mask/compact max error:   about 2.6e-15
bounded/exact max error:  about 2.05
```

These are an executed local float64 example and counted query rows, **not a 4x speedup**. The first error checks two implementations of the same bounded graph. The second deliberately shows that the bounded graph is not the exact graph. Changing the seed or keep length changes that error; it is not a model-quality metric.

The full encoder source remains available to consumers in both bounded paths. Compacting queries is not equivalent to deleting all prefix data.

## What is implemented

| Experiment | Implemented behavior | Deliberate simplification |
| --- | --- | --- |
| Replay | Causal encoder, shared source KV, layered SWA consumers; exact, bounded mask and compact forwards; incremental decode oracle | Single head, NumPy, no MLA/MoE/quantization/CP; no trained DeepSeek architecture |
| Speculative verification | Real target/draft forwards, greedy prefix acceptance, first rejection, bonus, EOS, committed-state comparison | Recomputes full target prefixes as an oracle; no probabilistic rejection sampling or speed claim |
| State | Exact prefix/boundary/layout keys, independent TTL caches, deep-copy isolation, rollback after ring overwrite | In-process arrays and fake clock; no network, persistence or memory-pressure control |
| Budget | Cumulative survival probabilities, discrete graph costs, total-budget search and newer-confidence redistribution | Caller-provided confidence and synthetic curve; no confidence head, CUDA Graph or asynchronous execution |

The ring rollback and TTL are isolated state experiments. They are not secretly wired into the speculative generator, which rebuilds its committed branch. Keeping that distinction visible is part of the point.

## Read it in this order

1. [`model.py`](core_deepinfer/model.py): full versus incremental causal computation, source KV and local consumer KV.
2. [`replay.py`](core_deepinfer/replay.py): query layout, placeholder rows and the numerical comparison.
3. [`speculative.py`](core_deepinfer/speculative.py): check which tokens may commit, independently of how quickly they were proposed.
4. [`cache.py`](core_deepinfer/cache.py): observe overwritten slots before and after rollback; expire short state without evicting the prefix.
5. [`budget.py`](core_deepinfer/budget.py): separate useful token budget from graph padding and try a different cost curve.

The [source map](docs/SOURCE_MAP.md) links each idea to a pinned upstream implementation and lists what this project does differently. Longer source walkthroughs are in [AI Infra Notes](https://github.com/he-yufeng/ai-infra-notes).

## Small changes worth trying

Increase decoder depth while keeping the local window fixed and compare bounded recovery with the exact oracle. Then remove the keep lower bound from just one local mask: the mask/compact test should catch the semantic change.

Give both requests high confidence but make the next graph much more expensive. The best useful budget need not grow just because more drafts are likely to be accepted.

Finally, replace ring snapshot restoration with only restoring the cursor. The overwrite test should fail because the old values have already been lost.

## Validation and limits

The current source tests cover numerical alignment across multiple seeds, future-token causality, a bounded-versus-exact counterexample, greedy verification, exhaustive budget comparisons, TTL boundaries and overwritten-slot rollback. Exact commands and the tested environment are recorded in [VALIDATION.md](docs/VALIDATION.md).

This project does not claim to reproduce V4.1 quality, DSpark's learned confidence, GPU scheduling overlap, 3FS, Mooncake, or production throughput. It has no server and executes no tools or shell commands from model output.

Original implementation and documentation: [MIT](LICENSE). Referenced upstream code retains its own licensing and attribution. Not affiliated with or endorsed by DeepSeek.

If a runnable source companion is useful to you, a Star is welcome. Issues with a small counterexample are especially helpful.
