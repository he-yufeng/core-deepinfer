"""In-process reference for exact state keys, independent TTLs and safe rollback."""

import hashlib
import math
import struct
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
from time import monotonic
from uuid import uuid4

import numpy as np

from .model import positive_int


@dataclass(frozen=True)
class CacheKey:
    model: str
    layout: str
    prefix_hash: str
    boundary: int


def cache_key(model, layout, tokens, *, alignment=1):
    positive_int(alignment, "alignment")
    tokens = tuple(tokens)
    if not model or not layout or not tokens or len(tokens) % alignment:
        raise ValueError("a state key requires a model, layout and aligned nonempty prefix")
    if any(type(t) is not int or not 0 <= t < 2**63 for t in tokens):
        raise ValueError("prefix tokens must be nonnegative int64 values")
    digest = hashlib.sha256()
    for token in tokens:
        digest.update(struct.pack("<q", token))
    return CacheKey(model, layout, digest.hexdigest(), len(tokens))


class StateCache:
    def __init__(self, *, capacity=64, clock=monotonic):
        self.capacity = positive_int(capacity, "capacity")
        self.clock = clock
        self.prefixes = OrderedDict()
        self.states = OrderedDict()

    def _put(self, table, key, value, ttl):
        ttl = float(ttl)
        if not math.isfinite(ttl) or ttl <= 0:
            raise ValueError("ttl must be finite and positive")
        entry = (float(self.clock()) + ttl, deepcopy(value))
        table.pop(key, None)
        while len(table) >= self.capacity:
            table.popitem(last=False)
        table[key] = entry

    def _get(self, table, key):
        entry = table.get(key)
        if entry is None:
            return None
        expiry, value = entry
        if float(self.clock()) >= expiry:
            del table[key]
            return None
        table.move_to_end(key)
        return deepcopy(value)

    def put_prefix(self, key, value, *, ttl):
        self._put(self.prefixes, key, value, ttl)

    def put_state(self, key, value, *, ttl):
        self._put(self.states, key, value, ttl)

    def get_prefix(self, key):
        return self._get(self.prefixes, key)

    def get_state(self, key):
        return self._get(self.states, key)


@dataclass(frozen=True)
class RingSnapshot:
    owner: str
    values: np.ndarray
    positions: np.ndarray
    next_position: int


class StateRing:
    def __init__(self, capacity, width):
        self.capacity = positive_int(capacity, "capacity")
        self.width = positive_int(width, "width")
        self.values = np.zeros((capacity, width), dtype=np.float64)
        self.positions = np.full(capacity, -1, dtype=np.int64)
        self.next_position = 0
        self._owner = uuid4().hex

    def append(self, value):
        value = np.asarray(value, dtype=np.float64)
        if value.shape != (self.width,) or not np.isfinite(value).all():
            raise ValueError("ring value must be a finite vector with the configured width")
        position = self.next_position
        slot = position % self.capacity
        self.values[slot] = value
        self.positions[slot] = position
        self.next_position += 1

    def read(self, position):
        if type(position) is not int or not 0 <= position < self.next_position:
            raise KeyError(position)
        slot = position % self.capacity
        if self.positions[slot] != position:
            raise KeyError(position)
        return self.values[slot].copy()

    def checkpoint(self):
        return RingSnapshot(self._owner, self.values.copy(), self.positions.copy(), self.next_position)

    def rollback(self, snapshot):
        if snapshot.owner != self._owner:
            raise ValueError("checkpoint belongs to another ring")
        self.values[:] = snapshot.values
        self.positions[:] = snapshot.positions
        self.next_position = snapshot.next_position
