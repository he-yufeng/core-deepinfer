# Validation

The checked-in [CPU experiment results](results/cpu-experiments.json) and [validation receipt](results/validation.json) come from running this implementation, not DeepInfer kernels or DeepSeek weights. The receipt records hashes of the Python modules, tests, build configuration and CI definition so that the tested source can be identified without depending on a mutable branch name.

## Reproduce locally

Use Python 3.10 or newer in a fresh environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
ruff check core_deepinfer tests
ruff format --check core_deepinfer tests
python -m pytest tests -q
python -m core_deepinfer all --json
python -m build
```

The recorded local run used Python 3.14.5, NumPy 2.5.4, pytest 9.1.1, Ruff 0.17.1 and Hatchling 1.32.4 on macOS arm64. All **84 tests passed**, lint and formatting passed, and the sdist and wheel built successfully. The wheel was also installed and exercised outside the checkout; isolated imports resolved to the environment's `site-packages`, not the source tree. These are local results. The [CPU workflow](../.github/workflows/tests.yml) independently checks Python 3.10, 3.12 and 3.14 on Linux; its current status is visible in [GitHub Actions](https://github.com/he-yufeng/core-deepinfer/actions/workflows/tests.yml).

## What the checks establish

| Area | Check | What it does not establish |
| --- | --- | --- |
| Causal model | Incremental steps match full exact forwards across three seeds; changing future tokens preserves earlier logits | Compatibility with trained DeepSeek checkpoints |
| Bounded replay | Nine seed/shape cases compare mask and compact logits and local keys at absolute tolerance `1e-10`; complete encoder source is retained | Equivalence of bounded replay and exact recovery |
| Counterexample | A fixed bounded case differs from the exact oracle while its mask and compact paths agree | A task-quality estimate or an acceptable production error threshold |
| Greedy verification | Three seeds and three draft widths match ordinary generation; explicit logits cover first/middle rejection, all-accepted bonus and EOS | DSpark sampling, learned confidence or faster inference |
| State | Exact key identity, TTL boundaries, copy isolation and restoration of overwritten ring contents; checkpoints survive a cache round trip | Concurrency, persistence, distributed leases or memory-pressure handling |
| Budget | Four random confidence cases match exhaustive allocation search; redistribution conserves useful budget; graph padding stays separate | The accuracy of a real hardware cost model |
| Packaging | Build from sdist, wheel installation, isolated import and outside-checkout CLI smoke checks; archive contents exclude local environments and private artifacts | A PyPI release or a hosted service |

Additional regressions cover a single-pass draft iterator, low-precision NumPy TTL input and unsigned NumPy chunk indices. They matter because ordinary Python lists, floats and integers would not expose those failure paths.

## Reading the default results

For `python -m core_deepinfer all --json` with the default seed and replay settings:

- Replay processes 96 consumer query rows in mask mode and 24 in compact mode. Both retain 32 encoder source rows. The local maximum mask/compact logit difference is about `2.6e-15`; bounded/exact differs by about `2.05`.
- The budget example chooses four useful positions, with request lengths `[3, 1]`. Reversing the confidence input redistributes them to `[1, 3]`. Its `1.7 ms` estimate comes entirely from the supplied synthetic curve and sampler cost.
- The cache example expires short state at its TTL without evicting the longer-lived prefix, then restores the three original ring values after overwrite.
- Speculative and ordinary greedy generation produce the same 16 tokens. The default random draft accepts zero of 70 proposed tokens. This is a correctness example, not a useful language response or an acceptance-rate benchmark. Separate tests exercise successful prefix acceptance.

Small floating-point differences across NumPy versions and platforms are expected; compare against the stated tolerance, not the last printed digit. A reduced consumer row count is not a measured end-to-end speedup: encoder work, complete source storage and other operations remain.

## Inspect an installed wheel

After building, install the wheel in an environment with NumPy and run from another directory:

```bash
python -m pip install --no-deps dist/core_deepinfer-0.1.0-py3-none-any.whl
# Change to a directory outside this checkout before running these commands.
python -I -c 'import core_deepinfer; print(core_deepinfer.__file__)'
python -I -m core_deepinfer all --json
core-deepinfer replay --length 24 --keep 4 --json
```

The import path should contain `site-packages`. No model download, API call or network service is part of any experiment.

No GPU/NPU, original native engine, model-quality evaluation or production throughput test was run for this project. Ring rollback and TTL remain independent demonstrations; the speculative generator rebuilds the committed branch rather than using a transactional ring cache.
