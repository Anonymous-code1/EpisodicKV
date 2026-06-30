"""Baselines: full attention and a static-kmeans (RetroInfer/ClusterKV-style) cache.

These share the SAME cluster-aggregate attention surrogate as EpisodicKV's
estimation path, so any accuracy/error difference is attributable purely to the
*partitioning* (episode-field vs plain key-similarity) and the *update policy*,
which is exactly the comparison the paper needs.
"""
from __future__ import annotations
import numpy as np


def full_attention(q: np.ndarray, K: np.ndarray, V: np.ndarray,
                   d: int | None = None, temp: float | None = None) -> np.ndarray:
    """Exact attention output o = softmax(qK^T/temp) V (temp defaults to sqrt(d))."""
    d = d or q.shape[-1]
    logits = (K @ q) / (temp if temp is not None else np.sqrt(d))
    logits -= logits.max()
    w = np.exp(logits)
    w /= w.sum()
    return w @ V


def _spherical_kmeans(X, k, iters=10, seed=0):
    from .episode_field import _spherical_kmeans as skm, _l2norm
    return skm(_l2norm(X), k, iters=iters, seed=seed)


class KMeansKVCache:
    """RetroInfer-style baseline: cluster keys by pure similarity (lambda=0),
    re-cluster on a *fixed schedule* (every `update_every` tokens).

    Same precise-retrieval + Jensen-estimation attention as EpisodicKV, so the
    only differences are (1) no temporal field, (2) blind update schedule.
    """

    def __init__(self, tokens_per_centroid: int = 16, kmeans_iters: int = 10,
                 update_every: int = 1024, seed: int = 0):
        self.tpc = tokens_per_centroid
        self.iters = kmeans_iters
        self.update_every = update_every
        self.seed = seed
        self.reset()

    def reset(self):
        self.K_buf: list[np.ndarray] = []
        self.V_buf: list[np.ndarray] = []
        self.mu: list[np.ndarray] = []
        self.size: list[int] = []
        self.vsum: list[np.ndarray] = []
        self.mem_keys: list[np.ndarray] = []
        self.mem_vals: list[np.ndarray] = []
        self._since_update = 0

    def append(self, k: np.ndarray, v: np.ndarray):
        self.K_buf.append(k)
        self.V_buf.append(v)
        self._since_update += 1
        if self._since_update >= self.update_every:
            self._recluster()
            self._since_update = 0

    def _recluster(self):
        if not self.K_buf:
            return
        K = np.stack(self.K_buf)
        V = np.stack(self.V_buf)
        n = K.shape[0]
        k = max(1, n // self.tpc)
        labels = _spherical_kmeans(K, k, iters=self.iters, seed=self.seed)
        self.mu, self.size, self.vsum = [], [], []
        self.mem_keys, self.mem_vals = [], []
        for c in np.unique(labels):
            mask = labels == c
            self.mu.append(K[mask].mean(axis=0))
            self.size.append(int(mask.sum()))
            self.vsum.append(V[mask].sum(axis=0))
            self.mem_keys.append(K[mask].copy())
            self.mem_vals.append(V[mask].copy())

    def finalize(self):
        """Force a final clustering of any remaining buffered tokens."""
        self._recluster()

    def attention(self, q: np.ndarray, steady_keys, steady_vals,
                  r_clusters: int, d: int | None = None, temp: float | None = None):
        from .retrieval import episode_decayed_attention

        outer = self

        class _Idx:  # adapt to retrieval API
            def __init__(s):
                s._mu = np.stack(outer.mu) if outer.mu else np.zeros((0, q.shape[-1]))
                s._s = np.asarray(outer.size, float)
                s._vs = np.stack(outer.vsum) if outer.vsum else np.zeros((0, q.shape[-1]))
                s.mem_keys = outer.mem_keys
                s.mem_vals = outer.mem_vals
            def centroids(s): return s._mu
            def sizes(s): return s._s
            def value_sums(s): return s._vs
            def recency(s): return np.zeros(len(outer.mu))
        out = episode_decayed_attention(
            q, _Idx(), steady_keys, steady_vals, r_clusters,
            gamma=0.0, eta=0.0, d=d, temp=temp, return_diagnostics=False)
        return out  # ndarray


class SlidingWindowKVCache:
    """Fixed temporal segments (no semantic awareness)."""

    def __init__(self, window: int = 1024, tokens_per_centroid: int = 16,
                 kmeans_iters: int = 10, seed: int = 0):
        self.window = window
        self.tpc = tokens_per_centroid
        self.iters = kmeans_iters
        self.seed = seed
        self.reset()

    def reset(self):
        self.K_buf: list[np.ndarray] = []
        self.V_buf: list[np.ndarray] = []
        self.mu: list[np.ndarray] = []
        self.size: list[int] = []
        self.vsum: list[np.ndarray] = []
        self.mem_keys: list[np.ndarray] = []
        self.mem_vals: list[np.ndarray] = []
        self.episode_id: list[int] = []
        self._seg_tokens = 0
        self._next_ep = 0

    def append(self, k: np.ndarray, v: np.ndarray):
        self.K_buf.append(k)
        self.V_buf.append(v)
        self._seg_tokens += 1
        if self._seg_tokens >= self.window:
            self._flush_segment()
            self._seg_tokens = 0

    def _flush_segment(self):
        if not self.K_buf:
            return
        K = np.stack(self.K_buf)
        V = np.stack(self.V_buf)
        n = K.shape[0]
        k = max(1, n // self.tpc)
        labels = _spherical_kmeans(K, k, iters=self.iters, seed=self.seed)
        for c in np.unique(labels):
            mask = labels == c
            self.mu.append(K[mask].mean(axis=0))
            self.size.append(int(mask.sum()))
            self.vsum.append(V[mask].sum(axis=0))
            self.mem_keys.append(K[mask].copy())
            self.mem_vals.append(V[mask].copy())
            self.episode_id.append(self._next_ep)
        self._next_ep += 1
        self.K_buf, self.V_buf = [], []

    def finalize(self):
        self._flush_segment()

    def attention(self, q, steady_keys, steady_vals, r_clusters, d=None, temp=None):
        from .retrieval import episode_decayed_attention

        outer = self

        class _Idx:
            def centroids(s): return np.stack(outer.mu) if outer.mu else np.zeros((0, q.shape[-1]))
            def sizes(s): return np.asarray(outer.size, float)
            def value_sums(s): return np.stack(outer.vsum) if outer.vsum else np.zeros((0, q.shape[-1]))
            def recency(s): return np.zeros(len(outer.mu))
            def episodes(s): return np.asarray(outer.episode_id, dtype=np.int64)
            @property
            def mem_keys(s): return outer.mem_keys
            @property
            def mem_vals(s): return outer.mem_vals

        return episode_decayed_attention(
            q, _Idx(), steady_keys, steady_vals, r_clusters,
            gamma=0.0, eta=0.0, d=d, temp=temp, rho=0.0)


class FixedIntervalEpisodeCache(SlidingWindowKVCache):
    """Fixed-length episode segments with episode-field clustering."""

    def __init__(self, episode_len: int = 120, lam: float = 0.8, **kw):
        super().__init__(window=episode_len, **kw)
        self.lam = lam
        self.r = 8
        self.episode_scale = 1024.0
        self._positions: list[int] = []
        self._pos = 0

    def reset(self):
        super().reset()
        self._positions = []
        self._pos = 0

    def append(self, k: np.ndarray, v: np.ndarray):
        self.K_buf.append(k)
        self.V_buf.append(v)
        self._positions.append(self._pos)
        self._pos += 1
        self._seg_tokens += 1
        if self._seg_tokens >= self.window:
            self._flush_segment()
            self._seg_tokens = 0

    def _flush_segment(self):
        if not self.K_buf:
            return
        from .episode_field import EpisodeFieldIndex
        idx = EpisodeFieldIndex(lam=self.lam, r=self.r,
                                episode_scale=self.episode_scale,
                                tokens_per_centroid=self.tpc,
                                kmeans_iters=self.iters, seed=self.seed)
        K = np.stack(self.K_buf)
        V = np.stack(self.V_buf)
        P = np.asarray(self._positions)
        idx.add_segment(K, V, P)
        for i in range(idx.num_clusters):
            self.mu.append(idx.mu[i])
            self.size.append(idx.size[i])
            self.vsum.append(idx.vsum[i])
            self.mem_keys.append(idx.mem_keys[i])
            self.mem_vals.append(idx.mem_vals[i])
            self.episode_id.append(self._next_ep)
        self._next_ep += 1
        self.K_buf, self.V_buf, self._positions = [], [], []


class RoPETemporalKMeansCache(KMeansKVCache):
    """RoPE temporal weighting in clustering only (no adaptive boundary, no EER)."""

    def __init__(self, lam: float = 0.8, r: int = 8, episode_scale: float = 1024.0, **kw):
        super().__init__(**kw)
        self.lam = lam
        self.r = r
        self.episode_scale = episode_scale
        self._positions: list[int] = []
        self._pos = 0

    def reset(self):
        super().reset()
        self._positions = []
        self._pos = 0

    def append(self, k: np.ndarray, v: np.ndarray):
        self.K_buf.append(k)
        self.V_buf.append(v)
        self._positions.append(self._pos)
        self._pos += 1
        self._since_update += 1
        if self._since_update >= self.update_every:
            self._recluster()
            self._since_update = 0

    def _recluster(self):
        if not self.K_buf:
            return
        from .episode_field import EpisodeFieldIndex
        idx = EpisodeFieldIndex(lam=self.lam, r=self.r,
                                episode_scale=self.episode_scale,
                                tokens_per_centroid=self.tpc,
                                kmeans_iters=self.iters, seed=self.seed)
        K = np.stack(self.K_buf)
        V = np.stack(self.V_buf)
        P = np.asarray(self._positions)
        phi = idx.field(K, P)
        k = max(1, K.shape[0] // self.tpc)
        labels = _spherical_kmeans(phi, k, iters=self.iters, seed=self.seed)
        self.mu, self.size, self.vsum = [], [], []
        self.mem_keys, self.mem_vals = [], []
        for c in np.unique(labels):
            mask = labels == c
            self.mu.append(K[mask].mean(axis=0))
            self.size.append(int(mask.sum()))
            self.vsum.append(V[mask].sum(axis=0))
            self.mem_keys.append(K[mask].copy())
            self.mem_vals.append(V[mask].copy())