# 06 Fork a working core, not a feature list

[中文](06-fork-and-extend.md) · [Series](00-index_EN.md)

A useful reference lets you change one decision and observe its consequence through the complete loop, with an independent oracle that can identify the mistake. Protect readable size without treating a smaller line count as the goal.

Run [`examples/engine_batch.py`](../../examples/engine_batch.py). It submits requests through the library, cancels one, steps a round, drains the rest and reuses the same engine. Healthy outputs are checked against ordinary greedy generation. Embedding the engine does not require going through its CLI.

## Start with scheduling

Keep the model fixed and lower `EngineConfig.max_batch_tokens` from eight to two for a mixed-length batch. Round counts and verification widths should change. Under default exact recovery, correct final tokens and state should not.

Then change `cost_curve`, making larger sizes expensive. Fewer useful drafts may be verified; padding must not become additional output. Without real hardware timing, that curve is an experimental input, not evidence that the engine found the fastest configuration.

## Change recovery policy

`_open` currently selects the longest stored prefix. Compare the work required by a shorter hot cursor and a longer warm encoder prefix, then write a different choice policy. The missing ingredient is a cost model and a useful comparison, not a more elaborate name for `hit=True`.

Compact recovery can legitimately change model trajectories. Do not loosen exact-mode regression tolerances to hide that semantic change. Keep implementation agreement within the bounded graph separate from the counterexample showing its difference from exact recovery.

## Reduce transaction copying

Copied checkpoints make `Runner` inspectable, but copying global source state at draft boundaries is not cheap. An extension could retain undo records for overwritten slots and advance source lengths only when appropriate.

Local keys and values, position tags, encoder/shared source state, token boundary and logits must remain coordinated. Restoring just one component is insufficient even if a few output IDs happen to match. Begin with an overwrite-sensitive failure and compare complete state before making a performance claim.

## A different model has a different state contract

`TinyModel` is not an arbitrary Hugging Face checkpoint wrapper. `Runner` depends on its causal encoder, shared source and per-layer local state. A model without those components needs a restated interface and tests. Renaming ordinary Llama inference does not reproduce DeepInfer.

Do not reuse old parameter or layout fingerprints either. Different weights, windows, layouts and recovery modes cannot share local state indiscriminately. A GPU backend would also need device and dtype validity, not merely a surviving Python object.

## Keep the omissions honest

There is no trained language capability, probabilistic rejection sampling, parallel GPU batching, CP, distributed KV storage, persistent session format or HTTP server. Each requires its own real validation. Copying another project's checklist does not complete this engine.

A useful extension leaves a working user entry point, a check sensitive to an incorrect implementation, and an accurate account of its simplifications. Protect readable size, but do not achieve it by deleting error handling.

```bash
python examples/engine_batch.py
python -m core_deepinfer generate --input examples/requests.json --trace --json
python -m pytest -q
```

Finally build, install, and run outside the checkout. Importing from the source directory does not prove that the wheel and console entry point work. [Validation](../../docs/VALIDATION.md) records those checks separately.
