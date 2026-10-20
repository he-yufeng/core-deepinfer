# 05 One model, two different equivalence questions

[中文](05-model-and-replay.md) · [Series](00-index_EN.md)

[`TinyModel`](../model.py) does not reproduce a production architecture. Single-head float64 attention, one causal encoder, layered local consumers and shared source KV are enough to create real dependencies and state. Its random, untrained parameters are not a basis for judging task quality.

## Input and prediction positions are offset

The last logit row from `prefill([1, 2, 3])` predicts the next token. `step(state, token)` first consumes that token, then predicts the following one. Speculative verification already has its first distribution in the prefix cursor; consuming a draft before judging it against that distribution is an off-by-one semantic error.

Shapes can remain perfectly valid while acceptance becomes wrong. Having both a target object and a draft object is not sufficient evidence of correct verification.

## Three kinds of history

Encoder keys and values preserve causal history for the next encoder position. Encoder outputs are projected into shared keys and values read by each consumer's cross attention. Every consumer also has its own local K/V window.

Incremental execution projects only the new source row and appends shared K/V instead of projecting the entire history again. Local K/V passes through real rings. The shared source remains complete; a small local window does not make that source disposable.

Tests compare full and incremental forwards, then change future input tokens and verify earlier outputs stay unchanged. Final argmax equality alone is weak: different distributions can temporarily choose the same token.

## Question one: did compaction change the same graph?

Bounded recovery raises the local-key lower bound to `length - keep`. Mask mode retains every query and changes visibility. Compact mode actually executes suffix queries only. Both use the same absolute positions, complete shared source and truncated local dependency graph.

Comparing retained logits and each layer's local K/V checks whether removing query rows introduced an additional difference.

```bash
python -m core_deepinfer replay --length 32 --keep 8 --seed 7 --json
```

Three consumers process 96 versus 24 query rows, but the encoder still processes 32 and cross attention still reads the complete source. That does not establish a fourfold model speedup.

## Question two: is the bounded graph still the exact graph?

Exact mode preserves the normal local dependencies. Bounded mode deliberately removes some. Layered propagation can make those graphs differ even when mask and compact agree almost perfectly within the bounded graph.

The example reports both mask/compact error and bounded/exact error. The first checks implementation agreement; the second demonstrates that approximation changed the result. Together they do not prove exact recovery.

`Engine` defaults to exact recovery. Compact recovery is explicit. Since this reference also contains a global encoder, the dependency halo of a purely local stack does not justify truncating encoder history.

```bash
python -m pytest tests/test_replay.py -q
python -m pytest tests/test_runtime.py -k 'replay or incremental' -q
```

In a separate copy, remove part of the shared source only in compact mode. You changed cross-attention visibility, not just the number of query rows. Even if emitted token IDs happen to stay the same, compare logits and complete state to expose the difference.
