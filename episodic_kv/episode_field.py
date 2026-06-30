"""Episode Field: the joint semantic x temporal index (Sec. 2.1 of PLAN.md).

The key departure from RetroInfer / ClusterKV: tokens are clustered in an
*augmented* coordinate phi_i = [ normalized_key ; lambda * temporal_phase(pos) ]
so that clusters are simultaneously semantically coherent AND temporally local,
i.e. they recover narrative *episodes* rather than order-free similarity blobs.
"""
from __future__ import annotations
import numpy as np


def _l2norm(x: np.ndarray, axis: int = -1, eps: float = 1e-8) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + eps)


def temporal_phase(positions: np.ndarray, r: int = 8, base: float = 10000.0,
                   episode_scale: float = 1024.0) -> np.ndarray:
    """Low-frequency phase embedding encoding *episode-scale* ordering.

    Uses the coarsest RoPE-style frequency bands so that nearby tokens share a
    phase (same episode) while far-apart tokens are pushed apart. `episode_scale`
    sets the wavelength of the lowest band ~ one episode.

    Returns (len(positions), r) with values in [-1, 1].
    """
    positions = np.asarray(positions, dtype=np.float64)
    # frequencies from ~episode_scale (low) downward; only low bands -> coarse time
    k = np.arange(r // 2)
    freqs = (1.0 / (base ** (2.0 * k / r))) / episode_scale  # very low freq
    ang = positions[:, None] * freqs[None, :]                # (n, r/2)
    emb = np.concatenate([np.sin(ang), np.cos(ang)], axis=-1)  # (n, r)
    return emb.astype(np.float64)


def _spherical_kmeans(X: np.ndarray, k: int, iters: int = 10,
                      seed: int = 0) -> np.ndarray:
    """Spherical k-means on (already unit-normalized rows). Returns labels."""
    n = X.shape[0]
    if k >= n:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    centers = X[rng.choice(n, size=k, replace=False)].copy()
    labels = np.zeros(n, dtype=np.int64)
    for _ in range(iters):
        sims = X @ centers.T                  # (n, k) cosine since unit rows
        labels = np.argmax(sims, axis=1)
        for c in range(k):
            mask = labels == c
            if mask.any():
                m = X[mask].mean(axis=0)
                nm = np.linalg.norm(m)
                if nm > 1e-8:
                    centers[c] = m / nm
    return labels


class EpisodeFieldIndex:
    """Episode-segmented vector index over a KV store.

    Maintains, per cluster c: centroid (in raw key space) mu_c, size s_c,
    value-sum VS_c, mean episode position (recency), and the raw member indices.
    Clustering happens in episode-field space; centroids exported in key space so
    attention scoring stays q . mu_c (inner product), as in RetroInfer.
    """

    def __init__(self, lam: float = 0.6, r: int = 8, episode_scale: float = 1024.0,
                 tokens_per_centroid: int = 16, kmeans_iters: int = 10,
                 seed: int = 0):
        self.lam = lam
        self.r = r
        self.episode_scale = episode_scale
        self.tpc = tokens_per_centroid
        self.kmeans_iters = kmeans_iters
        self.seed = seed
        self.reset()

    def reset(self):
        self.mu: list[np.ndarray] = []      # centroid in raw key space
        self.size: list[int] = []
        self.vsum: list[np.ndarray] = []    # sum of values
        self.pos: list[float] = []          # mean position (recency)
        self.members: list[np.ndarray] = [] # raw token indices
        self.episode_id: list[int] = []     # which episode segment a cluster is in
        self.mem_keys: list[np.ndarray] = []  # raw member keys (for precise retr.)
        self.mem_vals: list[np.ndarray] = []  # raw member values
        self.ep_coord: list[np.ndarray] = []  # cluster mean temporal-phase coord
        self._next_episode = 0

    # ----- episode-field embedding -----
    def field(self, keys: np.ndarray, positions: np.ndarray) -> np.ndarray:
        """phi = [ keyhat ; lambda * temporal_phase ] then unit-normalized."""
        kh = _l2norm(keys, axis=-1)
        tp = temporal_phase(positions, r=self.r, episode_scale=self.episode_scale)
        phi = np.concatenate([kh, self.lam * tp], axis=-1)
        return _l2norm(phi, axis=-1)

    # ----- build / append one episode segment -----
    def add_segment(self, keys: np.ndarray, values: np.ndarray,
                    positions: np.ndarray):
        """Cluster a window of tokens into NEW clusters (one episode segment).

        This is the incremental update primitive: never re-clusters old episodes.
        """
        n = keys.shape[0]
        if n == 0:
            return
        k = max(1, n // self.tpc)
        phi = self.field(keys, positions)
        labels = _spherical_kmeans(phi, k, iters=self.kmeans_iters, seed=self.seed)
        tp = temporal_phase(positions, r=self.r, episode_scale=self.episode_scale)
        for c in np.unique(labels):
            mask = labels == c
            mk = keys[mask]
            mv = values[mask]
            mp = positions[mask]
            self.mu.append(mk.mean(axis=0))
            self.size.append(int(mask.sum()))
            self.vsum.append(mv.sum(axis=0))
            self.pos.append(float(mp.mean()))
            self.members.append(mp.astype(np.int64))  # store original token positions
            self.mem_keys.append(mk.copy())
            self.mem_vals.append(mv.copy())
            self.ep_coord.append(tp[mask].mean(axis=0))   # mean temporal phase
            self.episode_id.append(self._next_episode)
        self._next_episode += 1

    # ----- exported arrays for retrieval -----
    @property
    def num_clusters(self) -> int:
        return len(self.mu)

    def centroids(self) -> np.ndarray:
        return np.stack(self.mu) if self.mu else np.zeros((0, 1))

    def sizes(self) -> np.ndarray:
        return np.asarray(self.size, dtype=np.float64)

    def value_sums(self) -> np.ndarray:
        return np.stack(self.vsum) if self.vsum else np.zeros((0, 1))

    def recency(self) -> np.ndarray:
        return np.asarray(self.pos, dtype=np.float64)

    def episodes(self) -> np.ndarray:
        return np.asarray(self.episode_id, dtype=np.int64)

    def cluster_episode_coord(self) -> np.ndarray:
        """Mean temporal-phase coordinate per cluster (episode position in the
        episode field). Used by Episode-Exclusive Retrieval."""
        return np.stack(self.ep_coord) if self.ep_coord else np.zeros((0, self.r))