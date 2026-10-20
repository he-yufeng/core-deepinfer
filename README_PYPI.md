# Core DeepInfer

**Follow a request from prefill to the last committed token.**

A small NumPy inference engine for reading, running and changing inference state transitions. Requests share one real execution path: queue, restore or prefill, propose drafts, allocate verification work, write KV, reject a branch, roll back and commit.

Inspired by [DeepInfer](https://github.com/deepseek-ai/DeepInfer) and the runnable-core approach of [CoreCoder](https://github.com/he-yufeng/CoreCoder). This is an independent community project, not an official DeepSeek package.

The model performs real float64 attention, projections and feed-forward operations with **random, untrained parameters**. It does not load DeepSeek weights or produce useful language. No GPU, model download, API key, server or network access is needed after installation.

## Install and run

Python 3.10 or newer:

```bash
python -m pip install core-deepinfer
core-deepinfer
core-deepinfer demo --trace --json
```

The default demo actually executes shared-prefix requests, a cancellation, speculative writes and rollback, hot state reuse, warm encoder-prefix recovery and a cold miss. It compares both output tokens and the complete committed state against ordinary greedy decoding. Its clock is injected; it does not sleep for expiry. Its cost curve is synthetic, not a hardware measurement.

## Read the included walkthroughs offline

The wheel contains an index and six chapters, each in English and Chinese. The CLI prints the Markdown directly; it does not open a browser or fetch content.

```bash
core-deepinfer guide --list
core-deepinfer guide 01
core-deepinfer guide 03 --lang zh
core-deepinfer guide 02 --lang en --json
```

The chapters follow a request's lifetime, hot and warm restoration, speculative transactions, verification budgets, the numerical model and extension exercises. Source links and test commands refer to the full source distribution; the installed wheel provides the executable modules and guide text, not a second copy of the test suite.

中文导读：`core-deepinfer guide --list --lang zh`。先运行完整请求链，再读状态恢复与回滚；这不是在 CPU 上运行 DeepSeek 训练权重。

## Use the Python library

```python
from core_deepinfer import Engine, EngineConfig, TinyModel

target = TinyModel(seed=7)
draft = TinyModel(seed=8, depth=1)
engine = Engine(target, draft, config=EngineConfig(max_batch_tokens=8))

engine.submit([1, 2, 3, 4], max_tokens=8)
engine.submit([1, 2, 3, 4, 5], max_tokens=5)
for result in engine.run():
    print(result.request_id, result.tokens, result.finish_reason)

# Keep the engine and cache for another request.
result = engine.generate([1, 2, 3, 4], max_tokens=4)
```

Use `step()` to inspect one round, `cancel(id)` between rounds, and `result(id)` to revisit a completed request. `run()` delivers new results once. Healthy requests drain before `RequestError.results` reports a batch containing failures; a reported old error does not poison later requests.

The default recovery mode is exact. Bounded mask and compact recovery are explicit approximations and can change output. Matching those two implementations on one bounded dependency graph does not prove equality with exact recovery.

## Inspect a different batch

```bash
core-deepinfer generate --prompt 1,2,3 --prompt 4,5,6 --max-tokens 6 --batch-budget 4 --json
core-deepinfer generate --prompt 1,2,3,4 --greedy --trace
core-deepinfer replay --length 32 --keep 8 --json
```

Inputs and outputs are token IDs. Query-row counters record actual work but are not FLOPs or speedup measurements. Requests execute sequentially in NumPy, not as a vectorized GPU batch. Draft self-probability is a scheduling heuristic, not DSpark's learned acceptance head.

## Get the complete source and tests

The source distribution includes the MIT license, modules, tests, examples, bilingual READMEs, walkthroughs and validation records. Download the source archive from the package's Files section, or use:

```bash
python -m pip download --no-deps --no-binary=:all: core-deepinfer==0.1.0
```

After extracting it, install `.[dev]` in a virtual environment and run `python -m pytest -q`. The checked-in validation record distinguishes source tests, wheel installation, controlled interruption checks and unexecuted remote CI.

Longer source readings of the original implementation are in [AI Infra Notes](https://github.com/he-yufeng/ai-infra-notes).

## Deliberate limits

No trained DeepSeek models, MLA, MoE, mHC, quantization, CP, CUDA Graphs, GPU/NPU backend, distributed KV store, persistent sessions or HTTP server. Snapshots copy arrays, source history grows, and completed requests remain inspectable in memory. Controlled interruption recovery is not persistence, thread safety or protection against process termination.

This reference makes inference state observable. It makes no production throughput, latency or model-quality claim.
