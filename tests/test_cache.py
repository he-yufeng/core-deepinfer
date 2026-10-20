import numpy as np
import pytest

from core_deepinfer.cache import StateCache, StateRing, cache_key


def test_low_precision_ttl_does_not_round_the_deadline_to_the_current_time():
    now = [1_000_000.0]
    cache = StateCache(clock=lambda: now[0])
    key = cache_key("a", "b", [1])
    cache.put_state(key, b"state", ttl=np.float32(0.01))
    assert cache.get_state(key) == b"state"
    now[0] += 0.02
    assert cache.get_state(key) is None


def test_short_state_expiry_does_not_evict_long_prefix():
    now = [0.0]
    cache = StateCache(clock=lambda: now[0])
    key = cache_key("model-a", "layout-1", [1, 2, 3, 4], alignment=4)
    cache.put_prefix(key, np.ones(3), ttl=100)
    cache.put_state(key, np.zeros(3), ttl=3)
    now[0] = 3.0
    assert cache.get_state(key) is None
    np.testing.assert_array_equal(cache.get_prefix(key), np.ones(3))
    now[0] = 100
    assert cache.get_prefix(key) is None


def test_model_layout_boundary_and_branch_identity_are_separate():
    base = cache_key("a", "x", [1, 2, 3, 4])
    others = [
        cache_key("b", "x", [1, 2, 3, 4]),
        cache_key("a", "y", [1, 2, 3, 4]),
        cache_key("a", "x", [1, 2, 3]),
        cache_key("a", "x", [1, 2, 3, 5]),
    ]
    assert all(key != base for key in others)


def test_cache_values_are_isolated_from_the_caller_and_other_branches():
    cache = StateCache()
    key = cache_key("a", "x", [1, 2])
    data = np.ones(2)
    cache.put_state(key, data, ttl=60)
    data[0] = 7
    restored = cache.get_state(key)
    restored[1] = 9
    np.testing.assert_array_equal(cache.get_state(key), np.ones(2))


def test_ring_rollback_restores_overwritten_values_not_just_the_cursor():
    ring = StateRing(capacity=3, width=2)
    for pos in range(3):
        ring.append([pos, pos + 1])
    checkpoint = ring.checkpoint()
    for pos in range(3, 6):
        ring.append([pos, pos + 1])
    with pytest.raises(KeyError):
        ring.read(0)
    ring.rollback(checkpoint)
    for pos in range(3):
        np.testing.assert_array_equal(ring.read(pos), [pos, pos + 1])
    with pytest.raises(KeyError):
        ring.read(3)
    ring.append([9, 10])
    np.testing.assert_array_equal(ring.read(3), [9, 10])


def test_ring_checkpoint_survives_a_cache_round_trip():
    ring = StateRing(1, 1)
    ring.append([1])
    cache = StateCache()
    key = cache_key("a", "ring", [1])
    cache.put_state(key, ring.checkpoint(), ttl=60)
    ring.append([2])
    ring.rollback(cache.get_state(key))
    np.testing.assert_array_equal(ring.read(0), [1])


def test_ring_rejects_checkpoint_from_another_request():
    first = StateRing(3, 2)
    second = StateRing(3, 2)
    with pytest.raises(ValueError):
        second.rollback(first.checkpoint())


@pytest.mark.parametrize("ttl", [0, -1, float("nan"), float("inf")])
def test_invalid_ttl_is_rejected(ttl):
    with pytest.raises(ValueError):
        StateCache().put_state(cache_key("a", "b", [1]), b"state", ttl=ttl)


def test_unaligned_state_boundary_is_rejected():
    with pytest.raises(ValueError):
        cache_key("a", "b", [1, 2, 3], alignment=4)
