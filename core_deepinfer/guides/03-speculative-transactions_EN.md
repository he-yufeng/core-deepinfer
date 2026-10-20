# 03 Writes that do not commit

[中文](03-speculative-transactions.md) · [Series](00-index_EN.md)

Final token equality can hide half of speculative correctness: after rejecting an output suffix, did the cache reject it too?

[`Runner`](../runtime.py) uses rings in the actual forward path. Each slot contains real local K/V across consumer layers. `advance` writes a new position, and the next forward reads through absolute position tags. With a small window, verifying one proposal can overwrite every old slot.

## Write first, then choose the commit boundary

The prefix cursor already has next-token logits. For k drafts, verification needs k+1 distributions: the existing one predicts the first draft, consuming that draft predicts the second, and the last distribution supplies a bonus.

`verify` saves the initial checkpoint, advances through every allocated draft and keeps intermediate checkpoints and logits. Those writes and computations really happened, whether or not their outputs eventually commit.

For a proposal `a, b, c, d`, suppose the target accepts `a` and rejects `b`. The output commits `a` and a target replacement, not `c` and `d`. State must restore the checkpoint after consuming `a`, then consume the replacement.

## More than a cursor

A checkpoint contains encoder history and shared source state, the token boundary, both local rings' contents and position tags, their write cursors, current logits, and a runner-specific owner ID.

Restoring only `next_position` cannot recover overwritten values. Restoring only local KV leaves rejected source positions visible to later cross attention. Restoring arrays without logits samples from the wrong next-token distribution.

This reference copies arrays to keep restoration inspectable. That is not an efficient GPU journal, and copying cost has not disappeared from the implementation.

## Keep target and draft at the same committed boundary

After target verification, the draft cursor may still point past the full proposal. `Engine.step` restores its checkpoint at the same accepted length, then consumes replacement or bonus tokens. Otherwise the next proposal starts from a suffix that the target never committed.

A forward failure restores the beginning of the current verification. It becomes an explicit request failure, not a partially written branch used by another request. Both new ring vectors are shape-checked before either write; validating the value only after writing the key leaves a different partial-state bug.

## Prove that verification is incremental

In [`test_engine.py`](../../tests/test_engine.py), `test_target_and_draft_prefill_only_once_not_on_each_verification` counts actual prefill calls. A separate runtime test forbids any further prefill after initialization, then verifies an all-accepted proposal and its bonus.

The numerical oracle checks more than emitted tokens: encoder history, projected shared keys and values, and every layer's local K/V must agree. It also checks the first new forward after restoration.

```bash
python -m pytest tests/test_runtime.py -q
python -m pytest tests/test_engine.py -k 'cached_speculative or prefill_only_once or identical_draft or eos' -q
python -m pytest tests/test_engine_failures.py -q
```

In a separate copy, change `StateRing.rollback` to restore only its cursor. Verify a proposal longer than window=1 or 2, and inspect the position-tag or numerical failure. This experiment tests whether writes can be undone, not whether a random draft has an impressive acceptance rate.
