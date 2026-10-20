# Validation

The checked-in [engine demo](results/engine-demo.json), [detail experiments](results/cpu-experiments.json) and [validation receipt](results/validation.json) record this CPU implementation, not DeepInfer kernels or DeepSeek weights. The receipt identifies tested modules, tests, guides and packaging inputs by SHA-256 rather than a mutable branch name.

## Reproduce

Use Python 3.10 or newer in a fresh environment, from the complete source distribution:

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
python -m core_deepinfer guide --list --json
python -m core_deepinfer guide 03 --lang zh
python -m build
```

The local environment is macOS arm64, Python 3.14.5, NumPy 2.5.4, pytest 9.1.1, Ruff 0.17.1 and Hatchling 1.32.4. **256 tests passed** in the source tree. These include 30 checks for the new offline guide interface and 21 regression cases from the interruption, configuration and parameter-identity audit. Test counts are coverage accounting, not a completeness or production-readiness claim.

The build explicitly selects standard Core Metadata 2.4 for both archives. Hatchling 1.32.4's default 2.5 output was rejected by the existing Twine 6.2.0 validator; this package needs no 2.5-only metadata fields. The configured version keeps the license expression and ordinary dependency metadata while remaining readable by that release toolchain.

The recorded local build uses existing dependencies with `python -m build --no-isolation`. This project's wheel is installed with `--no-deps`; the full copied test suite is then run outside the checkout against the installed package. Source and installed-wheel results are recorded separately. A new isolated editable dependency installation is not part of the successful local receipt. Package metadata is checked using an already-installed Twine 6.2.0; no new Twine installation is required.

The [CPU workflow](../.github/workflows/tests.yml) is configured for Python 3.10, 3.12 and 3.14 on Linux. A workflow definition is not a completed CI run; no remote run is claimed here. PyPI upload and remote file hashes, when available, are recorded in a separate publication receipt rather than inferred from a local build.

## What is checked

| Area | Actual checks | What they do not establish |
| --- | --- | --- |
| Model | Full and incremental causal forwards, future-token causality, cached shared projections, encoder-only replay, current-parameter fingerprints and stale-cursor rejection | DeepSeek checkpoint compatibility, concurrent weight mutation or trained language capability |
| Transactional runner | Actual KV rings, overwrite then rollback, middle/first rejection, all-accepted bonus, EOS, foreign checkpoint rejection, failure and controlled interruption restoration | Efficient GPU journaling, probabilistic rejection sampling or parallel verification |
| Request loop | Multiple requests, smaller-than-queue budgets, cancellation, healthy peers draining before error reporting, reusable engine, results delivered once, interrupted admission and commit recovery | An asynchronous server, persistence or vectorized GPU batching |
| Recovery | Cold/hot/warm/cold transitions, longest stored prefix extension, model/layout isolation, complete source retention, bounded cache capacity during interrupted insertion | Distributed caches, concurrency safety or a cost-optimal prefix policy |
| Budget | Rectangular and ragged exhaustive oracles, main-position admission, fixed-total newer-confidence redistribution that changes executed widths, padding separation, immutable normalized configuration | Calibrated acceptance confidence or a measured hardware curve |
| Differential scenarios | Twelve seeded mixed-lifecycle scenarios spanning three clock epochs, windows 1/2/4/8, small cache capacities, EOS and cancellation | Exhaustive testing of every possible configuration |
| CLI and package | Default demo, JSON and repeated-prompt requests, malformed input errors, library example, fourteen bundled guide texts, archive contents and outside-checkout wheel tests | A deployment, completed remote CI or support for untested platform combinations |

The default demo runs the real engine, not a scripted model. Its separate ordinary-greedy oracle compares tokens and every stored encoder/shared/local KV array. Oracle computation is excluded from the engine's query-row counters. Tests also count initial prefill calls and forbid re-prefilling during verification.

## Failure sensitivity

The overwrite tests exercise proposals longer than the ring window. A separate-copy mutation that restores only the ring cursor, leaving overwritten values and tags in place, must fail. The original working tree is not reverted or overwritten for that experiment. The receipt records the mutant separately; intentional failures are not passing product tests.

The audit reproduced these failures before fixing them:

- Catching only `Exception` left interrupted target or draft state ahead of committed output and could lose dequeued requests.
- Interrupting terminal cache insertion could lose completion or append another EOS on retry.
- An interrupted draft initialization could silently disable speculation for that request.
- Mutable external cost-curve rows could bypass validation or stop queue progress.
- A constructor-only parameter fingerprint let changed weights reuse old logits and continue from an incompatible cursor.
- Publishing a cache entry before eviction could leave its table over capacity; publishing a result before updating status could leave two conflicting notions of completion.

The regular regression suite includes controlled `KeyboardInterrupt` and `SystemExit` propagation, recovery of uncommitted peers, preservation of completed commits, model identity changes and both cache tables' insertion boundaries. Earlier shape, result-delivery and unsigned-index cases remain covered.

### Line-level fault injection

A separate deterministic fault driver injects one `KeyboardInterrupt` at a time in the observed engine, runtime and cache mutation paths. It covers up to the first five visits to each observed source line in two CPU scenarios: ordinary greedy execution and an identical-parameter draft, each with two requests and cache capacity one. It checks committed token/state alignment, terminal status, cache capacity, resumability and the final greedy oracle. The receipt gives the actual trial count and the exact module fingerprints.

This is not an exhaustive signal or concurrency proof. It does not inject every bytecode boundary, every occurrence or every rejection pattern. A second interruption during cleanup, process termination, power loss and concurrent engine use are outside its scope. The fault-driver trials are not added to the 256 pytest cases.

## Read the results correctly

The integrated demo uses an injected clock at 0, 3 and 200 seconds. It does not wait that long or measure latency. Its cost curve is synthetic, and draft self-probability is not a learned target-acceptance head.

The older `all` detail experiment reports 96 versus 24 consumer query rows for a 32-token replay case. Both paths retain the complete source. Near-zero mask/compact error checks the same bounded graph; its difference from exact recovery is intentional. None of these numbers proves a fourfold model speedup or a task-quality guarantee.

Check exact-mode generation against the stated absolute tolerance `1e-10`, not the last printed floating-point digit. Explicit bounded recovery can change subsequent tokens and is not covered by an exact-output claim.

## Installed-wheel check

After building in an environment with NumPy:

```bash
python -m pip install --force-reinstall --no-deps dist/core_deepinfer-0.1.0-py3-none-any.whl
# Move outside the checkout before the next commands.
python -I -c 'import core_deepinfer; print(core_deepinfer.__file__)'
python -I -m core_deepinfer demo --json
core-deepinfer generate --prompt 1,2,3 --max-tokens 4 --json
core-deepinfer guide --list --json
core-deepinfer guide 03 --lang zh
```

The isolated import must resolve under `site-packages`. The release check compares each installed module and guide with the source fingerprints, runs the complete copied tests outside the checkout, and compares installed demo JSON with source output.

The sdist includes examples, tests, validation records and the bilingual walkthroughs. The wheel includes executable modules and the same fourteen guide texts. Neither contains `.venv`, private artifacts, credentials or bytecode. The separate PyPI description uses no relative document links or nonexistent repository links.

No GPU/NPU, original native engine, trained model quality or production throughput validation is claimed. Checkpoints copy arrays, source arrays grow in memory, parameter fingerprints are recomputed, and the synchronous engine retains request history for inspection. It is a readable reference rather than a production memory manager.
