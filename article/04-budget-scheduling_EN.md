# 04 A fixed total can still move actual verification work

[中文](04-budget-scheduling.md) · [Series](00-index_EN.md)

A budget printed in a log is only an allocation example. In [`Engine.step`](../core_deepinfer/engine.py), follow `newer_lengths` into `proposal.tokens[:length - 1]` and then target verification. That slice makes the plan control execution.

## Reserve progress first

Each admitted request receives a main position. Length one means no draft verification, but the target can still commit its current prediction. Additional positions belong to drafts.

For R requests and k draft positions, useful budget is R+k. Admission cannot select more requests than its main positions can fit. The rest stay queued for a later round rather than receiving zero and getting stuck.

## Prefix probabilities, not independent prizes

Draft position j is useful only if its predecessors survive. `survival_probabilities` multiplies per-position inputs over the prefix. Ranked allocation must also preserve each request's prefix; a later position cannot be selected while its predecessor is absent.

The engine uses the draft's probability of its own chosen token. It is not calibrated or trained to estimate acceptance by the target. Calling a field confidence does not turn it into DSpark's learned confidence head.

## Choose a total, then update where it goes

With a cost curve, `choose_budget` scans useful totals, rounds each one to a sufficient discrete size and compares expected useful output with synthetic cost. Without a curve, it does not invent latency: it retains available work within the cap.

Previous-round probabilities can be stale relative to the current proposal. The engine uses them to select a total, then gives that same total to `reallocate` with newer probabilities. These are sequential reference calls, not simulated CPU/GPU overlap.

`test_new_confidence_redistribution_changes_executed_widths_not_just_a_printed_plan` injects controlled probabilities. A total of four stays four while lengths change from `[3, 1]` to `[1, 3]`. The test checks actual target cursor steps, not just allocation fields.

## Ragged availability is not matrix padding

One request may have only one output slot left while another has eight. A rectangular probability matrix is a storage format; its padded zeros are not available draft positions.

`draft_limits` constrains both the total and ranked candidates. Applying it only after allocation can still spend budget on nonexistent drafts, especially in a no-curve fallback or under a different cost curve.

`graph_size` is not a useful-token total either. Here it describes a discrete cost-model size, not an executed CUDA Graph. Padding never becomes extra tokens written into target state.

```bash
python -m pytest tests/test_budget.py tests/test_budget_ragged.py -q
python -m pytest tests/test_engine.py -k 'redistribution or ragged or small_budget' -q
```

Keep the model and requests fixed, then make a larger discrete size more expensive. Inspect selected totals and actual verification steps instead of assuming higher confidence must always mean more verification. Without hardware timing, the supported conclusion is about allocation behavior, not throughput.
