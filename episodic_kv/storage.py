"""Dual-layer Episode storage pool (global public + session private).

Layer-2 of the VLDB system design: shared prefix episodes are indexed globally
and reused across sessions; per-session incremental episodes are private and GC'd
on session end.
"""
from __future__ import annotations
import hashlib
from typing import Any

import numpy as np

from .episode_field import EpisodeFieldIndex


def _hash_keys(keys: np.ndarray) -> str:
    """Stable hash of a key sequence for prefix-cache lookup."""
    return hashlib.sha256(keys.astype(np.float32).tobytes()).hexdigest()


class GlobalPublicEpisodePool:
    """Shared prefix episodes (system prompt, common knowledge). LRU eviction."""

    def __init__(self, max_entries: int = 32, **index_kw):
        self.max_entries = max_entries
        self.index_kw = index_kw
        self._table: dict[str, EpisodeFieldIndex] = {}
        self._lru: list[str] = []

    def lookup(self, keys: np.ndarray) -> EpisodeFieldIndex | None:
        h = _hash_keys(keys)
        if h in self._table:
            self._lru = [x for x in self._lru if x != h] + [h]
            return self._table[h]
        return None

    def store(self, keys: np.ndarray, values: np.ndarray,
              positions: np.ndarray) -> EpisodeFieldIndex:
        h = _hash_keys(keys)
        if h not in self._table:
            idx = EpisodeFieldIndex(**self.index_kw)
            idx.add_segment(keys, values, positions)
            self._table[h] = idx
            self._lru.append(h)
            while len(self._lru) > self.max_entries:
                old = self._lru.pop(0)
                self._table.pop(old, None)
        else:
            self._lru = [x for x in self._lru if x != h] + [h]
        return self._table[h]

    def num_entries(self) -> int:
        return len(self._table)


class SessionPrivateEpisodePool:
    """Per-session incremental episode chain; GC on session end."""

    def __init__(self, session_id: str, **index_kw):
        self.session_id = session_id
        self.index = EpisodeFieldIndex(**index_kw)
        self.boundaries: list[int] = []
        self._pos = 0

    def add_segment(self, keys: np.ndarray, values: np.ndarray,
                    positions: np.ndarray):
        self.index.add_segment(keys, values, positions)
        self.boundaries.append(self._pos + len(keys))
        self._pos += len(keys)

    def gc(self) -> dict[str, Any]:
        """Release session state; return stats for observability."""
        stats = {
            "session_id": self.session_id,
            "num_clusters": self.index.num_clusters,
            "num_boundaries": len(self.boundaries),
        }
        self.index.reset()
        self.boundaries.clear()
        self._pos = 0
        return stats


class DualLayerStorage:
    """Combines global public pool + session private pool."""

    def __init__(self, session_id: str = "default", **index_kw):
        self.public = GlobalPublicEpisodePool(**index_kw)
        self.private = SessionPrivateEpisodePool(session_id, **index_kw)
        self._shared_prefix_len = 0

    def attach_shared_prefix(self, keys: np.ndarray, values: np.ndarray,
                             positions: np.ndarray) -> EpisodeFieldIndex:
        """Register or reuse a shared prefix; returns the public index."""
        self._shared_prefix_len = len(keys)
        return self.public.store(keys, values, positions)

    def private_index(self) -> EpisodeFieldIndex:
        return self.private.index

    def end_session(self) -> dict[str, Any]:
        return self.private.gc()
