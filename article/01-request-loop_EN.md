# 01 A request's lifetime starts with the loop

[中文](01-request-loop.md) · [Series](00-index_EN.md)

Leave attention aside for a moment. Open [`engine.py`](../core_deepinfer/engine.py), submit two requests of different lengths, and allow only two useful positions per round. Some requests wait, others commit a token, and one can be cancelled before the next step. These states decide whether a model call should happen at all.

```python
from core_deepinfer import Engine, EngineConfig

engine = Engine(config=EngineConfig(max_batch_tokens=2, max_active=2))
a = engine.submit([1, 2, 3], max_tokens=3)
b = engine.submit([4, 5, 6], max_tokens=5)
print(engine.step())
engine.cancel(b)
print([(r.request_id, r.finish_reason) for r in engine.run()])
```

## Admission is not execution

`submit` validates token IDs, output length, EOS and request identity before changing the queue. There is no successful-looking path that admits an input and silently omits it later. Generated IDs must also skip names already chosen by callers; otherwise an unrelated old name can reject a perfectly valid request.

`_prepare` creates the model cursor only when a request is selected. A queued cancellation can therefore avoid model execution entirely. A running request can be cancelled between `step` calls. This synchronous implementation does not interrupt a NumPy matrix multiplication already in progress.

## A round must make progress

`step` selects only as many requests as can receive a main verification position. A length-one allocation still commits the target's current next-token prediction. Having no draft allowance must not turn into having no progress.

Unfinished requests return to the tail. When more requests are queued than fit, later rounds select different requests instead of letting one long request own the front. This is request interleaving, not vectorized GPU batching. Planning work together and executing it in parallel are distinct capabilities.

The normal path can be reduced to this skeleton:

```python
# A skeleton; Engine.step contains the actual cancellation and error paths.
prepare(request)
proposal = draft.propose(width)
lengths = plan_verification_work(proposals)
decision = target.verify(proposal.tokens[:length - 1], max_tokens=remaining)
synchronize_draft_with_committed_tokens(decision)
commit_or_finish(request, decision)
```

The other parts open each operation and return to this same commit point. There is no second, simpler execution path reserved for the demo.

## Completion and delivery are different events

`_results` stores named terminal results. `run` returns newly completed results and remembers which IDs it delivered. `result(id)` still exposes historical results.

Without that distinction, a natural implementation mistake makes an engine unusable after one failure: every later drain sees the same historical error and raises it again, even when the new request succeeded. `test_engine_reuse.py` triggers a controlled forward failure, restores the model and submits another request to the same engine.

Failure also must not shorten the result list. Healthy peers drain before `RequestError.results` reports the batch, including its failed items. A caller can identify the failed input rather than reconstructing which input disappeared.

## Check it

```bash
python -m pytest tests/test_engine.py -k 'small_budget or cancelled or failure' -q
python -m pytest tests/test_engine_failures.py tests/test_engine_reuse.py -q
```

In a separate copy, put unfinished requests back at the front instead of the tail. Run a long request alongside several short ones under a small budget. Inspect progress order and result completeness first; elapsed time alone does not establish scheduler quality or a performance improvement.
