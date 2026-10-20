# 02 Two cache hits can require different computation

[中文](02-state-recovery.md) · [Series](00-index_EN.md)

The default demo can report four reused positions for both a hot hit and a warm hit. The hot path resumes immediately; the warm path still executes consumers. Counting only matched tokens loses that distinction.

Follow [`Engine._open`](../engine.py) and distinguish two payloads. [`EncoderState`](../model.py) contains causal encoder history and projected shared source KV. [`CachedState`](../runtime.py) also contains each consumer's local KV and the current next-token logits.

## Different payloads, independent lifetimes

The longer-lived table stores encoder prefixes. The short-lived table stores complete cursors. Each table handles its own TTL and LRU, so local-state expiry does not evict the longer-lived prefix. Values are copied on both insertion and retrieval; changing one restored branch cannot mutate another request's cached state.

The demo uses 100 seconds and 3 seconds to expose the two expiry boundaries. Those are teaching inputs, not the retention values from DeepSeek's deployment report or recommended production settings.

## Start with identity, not presence

A recoverable state's key contains the parameter fingerprint, layout, prefix hash and exact boundary. The consumer layout also includes window, depth, width and recovery mode. Otherwise a bounded local state might be silently restored as an exact one.

The fingerprint reads current parameter contents, not just the constructor seed. After changing `model.head`, an old cursor must reject further execution and a new request must not reuse old logits. This tiny model accepts the CPU cost of recomputing hashes; it has no weight hot-reload or concurrent mutation protocol.

The full encoder source can be shared between recovery modes because it was not truncated. That does not make the local state interchangeable. A test saves an exact cursor, requests the same prefix with compact recovery, and checks that the engine replays from the encoder rather than accepting the wrong local hot hit.

## Three actual execution paths

`_open` searches stored prefix boundaries from longest to shortest. With a complete cursor, `Runner.from_cached` rebuilds position-tagged rings without running model query rows for the prefix.

With only an encoder payload, `TinyModel.replay` reuses that source and recomputes consumers. Its `encoder_rows` is zero, but its `consumer_rows` is not. With neither payload, prefill runs from the input.

A shorter matching prefix is extended through actual `advance` calls. This is an exact token-prefix match, not semantic similarity.

```python
from core_deepinfer import Engine

engine = Engine()
engine.generate([1, 2, 3, 4], max_tokens=0)
result = engine.generate([1, 2, 3, 4, 5, 6], max_tokens=1)
# Four positions are restored; two new input positions are actually computed.
```

## A hit is not free

A hot hit still copies data into a request-owned cursor. Complete source reads have memory cost, and this NumPy reference grows source arrays with concatenation. Zero encoder query rows does not mean zero cost.

The current policy chooses the longest stored prefix. It does not price a shorter hot cursor against a longer warm encoder payload. That is a concrete extension point, not evidence that maximizing matched length minimizes recovery time. A real routing contract needs state validity and reconstruction cost as well as a hit length.

```bash
python -m pytest tests/test_engine.py -k 'real_cache or longest or another_models or bounded_layout' -q
python -m pytest tests/test_cache.py tests/test_model_identity.py -q
```

Change only `state_ttl` for a fixed request sequence, and inspect restore events and consumer row counts. Keep the model and budget unchanged so that the observed difference belongs to recovery rather than another variable.
