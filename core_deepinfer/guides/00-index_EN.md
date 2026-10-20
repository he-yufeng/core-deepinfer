# Read inference through Core DeepInfer

[中文](00-index.md) · [Project](../../README.md)

Submit a request, inspect what committed, then follow the state used by the next forward. Pause at a commit boundary, compare the target and draft cursors, and check what the next model call actually reads.

These six walkthroughs follow the same `Engine`, `Runner` and `TinyModel`. The subject is not a collection of formulas but the lifetime of an actual request: admission, recovery, proposal, verification, rollback, commit and termination. The older isolated numerical examples remain useful for examining individual details.

## Run before reading

```bash
python -m pip install -e '.[dev]'
python -m core_deepinfer demo --trace
python examples/engine_batch.py
```

The demo uses real model forwards and KV rings, not scripted token playback. The parameters are random and untrained, so the token IDs have no language capability. Synthetic costs and an injected clock are not hardware measurements. We are checking state transitions, not claiming an inference speedup. See [validation](../../docs/VALIDATION.md) for the full scope.

A PyPI installation includes these texts for offline reading: `core-deepinfer guide --list`, then `core-deepinfer guide 03`. Source links and test commands below refer to the complete source distribution; the wheel does not duplicate the test directory.

## The six parts

| Part | Question to follow | Entry point |
| --- | --- | --- |
| [01 A request's lifetime](01-request-loop_EN.md) | Who enters, who makes progress, and when a caller receives a result | `Engine.submit/step/run` |
| [02 Hot state and longer-lived prefixes](02-state-recovery_EN.md) | Why two cache hits can require different computation | `Engine._open`, `TinyModel.replay` |
| [03 Writes that do not commit](03-speculative-transactions_EN.md) | How to restore an accepted prefix after speculative ring overwrite | `Runner.verify/rollback` |
| [04 What the budget controls](04-budget-scheduling_EN.md) | Which actual verification positions move while the total stays fixed | `choose_budget/reallocate`, `Engine.step` |
| [05 The model and two different equivalences](05-model-and-replay_EN.md) | How source, queries, local state and next-token logits line up | `TinyModel.prefill/step/replay` |
| [06 Forking the reference](06-fork-and-extend_EN.md) | Make one change, observe it, and keep the claim within its evidence | Library, CLI and oracle tests |

Each part has executable test commands and one small modification to try. Those exercises are not unfinished features. They expose where a broken invariant becomes observable. Use a separate copy for destructive experiments rather than restoring over uncommitted work.

The [source map](../../docs/SOURCE_MAP.md) pins the upstream implementations and distinguishes them from our simplifications. Longer readings of the original project live in [AI Infra Notes](https://github.com/he-yufeng/ai-infra-notes). A result from this CPU engine is not evidence that an upstream kernel or production deployment was tested.
