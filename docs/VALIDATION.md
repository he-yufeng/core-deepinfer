# Validation

The checked-in [engine demo](results/engine-demo.json), [detail experiments](results/cpu-experiments.json) and [validation receipt](results/validation.json) record this CPU implementation, not DeepInfer kernels or DeepSeek weights. The receipt identifies tested Python, example, build and workflow files by SHA-256 rather than a mutable branch name.

## Reproduce

Use Python 3.10 or newer in a fresh environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
ruff check core_deepinfer tests examples
ruff format --check core_deepinfer tests examples
python -m pytest tests -q
python -m core_deepinfer demo --trace --json
python examples/engine_batch.py
python -m core_deepinfer generate --input examples/requests.json --json
python -m build
```

The local environment is macOS arm64, Python 3.14.5, NumPy 2.5.4, pytest 9.1.1, Ruff 0.17.1 and Hatchling 1.32.4. **205 tests passed**. The recorded local package build uses existing build dependencies with `python -m build --no-isolation`, followed by a dependency-free installation of this project's own wheel. A new isolated editable dependency installation is not part of that local receipt. The source example and installed-module/console checks are recorded separately.

The [CPU workflow](../.github/workflows/tests.yml) is configured for Python 3.10, 3.12 and 3.14 on Linux. A workflow definition is not a completed CI run; remote results must be checked independently.

## What is checked

| Area | Actual checks | What they do not establish |
| --- | --- | --- |
| Model | Full and incremental causal forwards, future-token causality, cached shared projections, encoder-only replay | DeepSeek checkpoint compatibility or trained language capability |
| Transactional runner | Actual KV rings, overwrite then rollback, middle/first rejection, all-accepted bonus, EOS, foreign checkpoint rejection, failure restoration | Efficient GPU journaling, probabilistic rejection sampling or parallel verification |
| Request loop | Multiple requests, smaller-than-queue budgets, cancellation, errors after healthy peers drain, reusable engine after a reported failure, results delivered once | An asynchronous server or vectorized GPU batching |
| Recovery | Cold/hot/warm/cold transitions, longest stored prefix extension, model/layout isolation, complete source retention | Persistent or distributed caches, concurrency safety or a cost-optimal prefix policy |
| Budget | Rectangular and ragged exhaustive oracles, main-position admission, fixed-total newer-confidence redistribution that changes executed widths, padding separation | Calibrated acceptance confidence or a measured hardware curve |
| Differential scenarios | Twelve seeded mixed-lifecycle scenarios spanning three clock epochs, windows 1/2/4/8, small cache capacities, EOS and cancellation | Exhaustive testing of every possible configuration |
| CLI and package | Default integrated demo, JSON and repeated-prompt requests, malformed input errors, library example, build, archive contents and outside-checkout wheel entry points | A PyPI publication, deployment or completed remote CI |

The default demo runs the real engine, not a scripted model. Its separate ordinary-greedy oracle compares tokens and every stored encoder/shared/local KV array. Oracle computation is excluded from the engine's query-row counters. Tests also count initial prefill calls and forbid re-prefilling during verification.

## Failure sensitivity

The overwrite tests exercise proposals longer than the ring window. A separate-copy mutation that restores only the ring cursor, leaving overwritten values and tags in place, must fail. The original working tree is not reverted or overwritten for that experiment.

Regression cases include:

- Key and value shapes are checked before either ring write, so malformed output cannot leave one cursor advanced alone.
- A terminal cache-write error remains explicit without making emitted tokens disagree with committed state.
- Generated request names skip caller-selected names.
- A reported failure does not cause every later `run()` on the same engine to raise the old error again.
- Single-pass draft input, low-precision TTL and unsigned NumPy chunk indices retain their earlier boundary coverage.

The receipt records the mutation result separately from the normal green tests. A failing mutant is not counted as another passing product test.

## Read the results correctly

The integrated demo uses an injected clock at 0, 3 and 200 seconds. It does not wait that long or measure latency. Its cost curve is synthetic, and draft self-probability is not a learned target-acceptance head.

The older `all` detail experiment still reports 96 versus 24 consumer query rows for a 32-token replay case. Both paths retain the complete source. Near-zero mask/compact error checks the same bounded graph; its difference from exact recovery is intentional. None of these numbers proves a fourfold model speedup or a task-quality guarantee.

Check exact-mode generation against the stated absolute tolerance `1e-10`, not the last printed floating-point digit. Explicit bounded recovery can change subsequent tokens and is not covered by an exact-output claim.

## Installed-wheel check

After building in an environment with NumPy:

```bash
python -m pip install --no-deps dist/core_deepinfer-0.1.0-py3-none-any.whl
# Move outside the checkout before the next commands.
python -I -c 'import core_deepinfer; print(core_deepinfer.__file__)'
python -I -m core_deepinfer demo --json
core-deepinfer generate --prompt 1,2,3 --max-tokens 4 --json
```

The isolated import should resolve under `site-packages`. The sdist includes examples and the bilingual walkthroughs; the wheel includes the executable package. Neither includes `.venv`, private artifacts, credentials or bytecode.

No GPU/NPU, original native engine, trained model quality or production throughput validation is claimed. Checkpoints copy arrays, source arrays grow in memory, and the synchronous engine retains request history for inspection. It is a readable reference rather than a production memory manager.
