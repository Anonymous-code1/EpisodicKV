"""End-to-end EpisodicKV cache: the drop-in plug-and-play object.

Streaming protocol per decoding step:
  1. append(k, v)                  -> token enters steady (local) window
  2. attention(q)                  -> episode-decayed approximate attention
  3. internally: ConflictPotential.update(...) -> if boundary fired OR local
     window full, close the episode and add_segment(...) to the index.
"""
from __future__ import annotations
import numpy as np

from .episode_field import EpisodeFieldIndex
from .conflict import ConflictPotential
from .retrieval import episode_decayed_attention
from .scheduler import AdaptiveScheduler, EpisodicMode, ModeConfig


class EpisodicKVCache:
    def __init__(self,
                 dim: int,
                 lam: float = 0.6,
                 tokens_per_centroid: int = 16,
                 r_clusters: int = 8,
                  sink: int = 4,
                  local_window: int = 64,
                  min_episode: int = 64,
                  max_segment: int = 1024,
                  gamma: float = 0.5,
                  eta: float = 0.1,
                  rho: float = 0.0,
                  beta: float = 1.0,
                  kappa: float = 2.0,
                  episode_scale: float = 1024.0,
                  temp: float | None = None,
                  seed: int = 0,
                  adaptive: bool = True,
                  scheduler: AdaptiveScheduler | None = None):
        self.dim = dim
        self.temp = temp
        self.r_clusters = r_clusters
        self.sink = sink
        self.local_window = local_window
        self.min_episode = min_episode
        self.max_segment = max_segment
        self.gamma = gamma
        self.eta = eta
        self.rho = rho
        self.episode_scale = episode_scale
        self.adaptive = adaptive
        self.scheduler = scheduler or AdaptiveScheduler()
        self._mode = EpisodicMode.FULL
        self._mode_cfg = self.scheduler.config_for_mode(EpisodicMode.FULL)
        self.index = EpisodeFieldIndex(
            lam=lam, tokens_per_centroid=tokens_per_centroid,
            episode_scale=episode_scale, seed=seed)
        self.conflict = ConflictPotential(beta=beta, kappa=kappa)
        self.reset()

    def reset(self):
        self.index.reset()
        self.conflict.reset()
        self._pos = 0
        self._sink_k: list[np.ndarray] = []
        self._sink_v: list[np.ndarray] = []
        self._buf_k: list[np.ndarray] = []   # current open episode buffer
        self._buf_v: list[np.ndarray] = []
        self._buf_p: list[int] = []
        # per-cluster conflict, recomputed lazily
        self._cluster_conflict: np.ndarray | None = None
        self.boundaries: list[int] = []      # positions where episodes closed
        self.U_history: list[float] = []
        self._mode = EpisodicMode.FULL

    def _apply_scheduler(self):
        if not self.adaptive:
            return
        cfg = self.scheduler.config_for_length(self._pos)
        if cfg is self._mode_cfg and self._mode == self.scheduler.mode_for_length(self._pos):
            return
        self._mode = self.scheduler.mode_for_length(self._pos)
        self._mode_cfg = cfg
        self.index.kmeans_iters = cfg.kmeans_iters
        self.conflict.kappa = cfg.conflict_kappa
        if cfg.rho > 0:
            self.rho = cfg.rho

    def episodic_enabled(self) -> bool:
        return self._mode_cfg.enabled

    # ----- steady (sink + local) views -----
    def _steady(self):
        ks = self._sink_k + self._buf_k[-self.local_window:]
        vs = self._sink_v + self._buf_v[-self.local_window:]
        if ks:
            return np.stack(ks), np.stack(vs)
        return np.zeros((0, self.dim)), np.zeros((0, self.dim))

    def _per_cluster_conflict(self) -> np.ndarray:
        """U_c: drift of each cluster's centroid from the active (latest) episode
        centroid. Cheap proxy used in episode-decayed scoring."""
        m = self.index.num_clusters
        if m == 0:
            return np.zeros(0)
        C = self.index.centroids()
        eps = self.index.episodes()
        active = eps.max()
        # active-episode mean centroid
        act = C[eps == active].mean(axis=0)
        act = act / (np.linalg.norm(act) + 1e-8)
        Cn = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-8)
        return 1.0 - Cn @ act

    def append(self, k: np.ndarray, v: np.ndarray):
        if len(self._sink_k) < self.sink:
            self._sink_k.append(np.asarray(k, float))
            self._sink_v.append(np.asarray(v, float))
        self._buf_k.append(np.asarray(k, float))
        self._buf_v.append(np.asarray(v, float))
        self._buf_p.append(self._pos)
        self._pos += 1
        self._apply_scheduler()

    def _close_episode(self):
        if not self._buf_k:
            return
        K = np.stack(self._buf_k)
        V = np.stack(self._buf_v)
        P = np.asarray(self._buf_p)
        self.index.add_segment(K, V, P)
        self.boundaries.append(self._pos)
        self._buf_k, self._buf_v, self._buf_p = [], [], []
        self._cluster_conflict = None

    def active_episode_coord(self) -> np.ndarray:
        """Episode-field temporal-phase coordinate of the CURRENT position.
        Episode-Exclusive Retrieval suppresses clusters far from this coord."""
        from .episode_field import temporal_phase
        p = max(self._pos - 1, 0)
        return temporal_phase(np.array([p]), r=self.index.r,
                              episode_scale=self.episode_scale)[0]

    def attention(self, q: np.ndarray, maybe_update: bool = True) -> np.ndarray:
        q = np.asarray(q, float)
        steady_k, steady_v = self._steady()
        from .baselines import full_attention

        if not self.episodic_enabled():
            if steady_k.shape[0] == 0:
                return np.zeros(self.dim)
            return full_attention(q, steady_k, steady_v, d=self.dim, temp=self.temp)

        if self.index.num_clusters == 0:
            if maybe_update:
                U, fired = self._update_conflict({"attn_mass": np.array([1.0])})
                self.U_history.append(U)
                if (fired and len(self._buf_k) >= self.local_window) \
                        or len(self._buf_k) >= self.max_segment:
                    self._close_episode()
            if steady_k.shape[0] == 0:
                return np.zeros(self.dim)
            return full_attention(q, steady_k, steady_v, d=self.dim, temp=self.temp)

        Uc = self._per_cluster_conflict()
        o, diag = episode_decayed_attention(
            q, self.index, steady_k, steady_v, self.r_clusters,
            gamma=self.gamma, eta=self.eta, conflict_per_cluster=Uc,
            d=self.dim, temp=self.temp, rho=self.rho,
            active_episode_coord=self.active_episode_coord(),
            return_diagnostics=True)

        if maybe_update:
            U, fired = self._update_conflict(diag)
            self.U_history.append(U)
            if (fired and self._refractory_ok()) \
                    or len(self._buf_k) >= self.max_segment:
                self._close_episode()
        return o

    def _refractory_ok(self) -> bool:
        """Allow a boundary only if the current episode is at least a minimum
        length (refractory period). Prevents over-segmentation from transient
        spikes; the min length is one local window."""
        min_len = max(self.min_episode, 1)
        return len(self._buf_k) >= min_len and \
            (self._pos - (self.boundaries[-1] if self.boundaries else 0)) >= min_len

    def _update_conflict(self, diag):
        """Two-timescale boundary detector: key drift + optional attention jump."""
        buf = self._buf_k
        short_w = self.local_window
        long_w = 4 * self.local_window
        if len(buf) >= 2:
            s = np.stack(buf[-short_w:]).mean(axis=0)
            l = np.stack(buf[-long_w:]).mean(axis=0)
            s /= (np.linalg.norm(s) + 1e-8)
            l /= (np.linalg.norm(l) + 1e-8)
            drift = 1.0 - float(s @ l)
        else:
            drift = 0.0

        if self._mode_cfg.use_attention_jump and self.index.num_clusters > 0 \
                and diag.get("attn_mass") is not None:
            ids = np.arange(self.index.num_clusters)
            cent = self.index.centroids()
            sz = self.index.sizes()
            mass = diag["attn_mass"]
            if mass.size == self.index.num_clusters:
                return self.conflict.update(ids, cent, sz, mass)

        return self.conflict.update_scalar(drift)

    def finalize(self):
        """Flush any open episode buffer into the index."""
        self._close_episode()