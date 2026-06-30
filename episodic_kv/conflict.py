"""Semantic Conflict Potential U(t) and the adaptive, parameter-free trigger.

Sec. 2.2 of PLAN.md. Replaces RetroInfer's blind "re-cluster every 1024 tokens"
with a principled, narrative-driven signal: update the index *when the episode
changes*, detected from (a) centroid semantic drift and (b) attention-mass jumps.
"""
from __future__ import annotations
import numpy as np


def _js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """Jensen-Shannon divergence between two distributions (aligned, same len)."""
    p = np.asarray(p, dtype=np.float64) + eps
    q = np.asarray(q, dtype=np.float64) + eps
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    kl = lambda a, b: np.sum(a * np.log(a / b))
    return float(0.5 * kl(p, m) + 0.5 * kl(q, m))


class ConflictPotential:
    """Tracks U(t) over the active window and fires episode boundaries.

    U(t) = sum_c w_c (1 - cos(mu_c^t, mu_c^{t-dt}))         # semantic drift
           + beta * JS(P_t || P_{t-dt})                      # attention-mass jump
    Trigger when U(t) > mean_t + kappa * std_t  (EWMA-based, no hand-tuned tau).
    """

    def __init__(self, beta: float = 1.0, kappa: float = 2.0, rho: float = 0.1,
                 warmup: int = 8):
        self.beta = beta
        self.kappa = kappa
        self.rho = rho
        self.warmup = warmup
        self.reset()

    def reset(self):
        self._prev_centroids: dict[int, np.ndarray] = {}  # episode/cluster id -> mu
        self._prev_mass: np.ndarray | None = None
        self._prev_ids: np.ndarray | None = None
        self.ewma_mean = 0.0
        self.ewma_var = 1e-6
        self.n_seen = 0
        self.history: list[float] = []

    @staticmethod
    def _cos(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        return float(a @ b / (na * nb + eps))

    def update_scalar(self, U: float) -> tuple[float, bool]:
        """Run the adaptive (EWMA mean+var) threshold on a precomputed scalar
        conflict potential U(t). Returns (U, boundary_fired)."""
        U = float(U)
        self.history.append(U)
        fired = False
        if self.n_seen >= self.warmup:
            tau = self.ewma_mean + self.kappa * np.sqrt(self.ewma_var)
            fired = U > tau
        prev_mean = self.ewma_mean
        self.ewma_mean = (1 - self.rho) * self.ewma_mean + self.rho * U
        self.ewma_var = (1 - self.rho) * self.ewma_var + self.rho * (U - prev_mean) ** 2
        self.n_seen += 1
        return U, bool(fired)

    def update(self, cluster_ids: np.ndarray, centroids: np.ndarray,
               sizes: np.ndarray, attn_mass: np.ndarray) -> tuple[float, bool]:
        """Feed the current step. Returns (U_t, boundary_fired)."""
        # (a) centroid semantic drift, matched by cluster id
        drift = 0.0
        wsum = 0.0
        for cid, mu, sz in zip(cluster_ids, centroids, sizes):
            if cid in self._prev_centroids:
                d = 1.0 - self._cos(mu, self._prev_centroids[cid])
                drift += sz * d
                wsum += sz
        drift = drift / wsum if wsum > 0 else 0.0

        # (b) attention-mass jump (align on shared cluster ids)
        jump = 0.0
        if self._prev_mass is not None and self._prev_ids is not None:
            common = np.intersect1d(cluster_ids, self._prev_ids)
            if common.size >= 2:
                cur = {int(c): m for c, m in zip(cluster_ids, attn_mass)}
                prv = {int(c): m for c, m in zip(self._prev_ids, self._prev_mass)}
                p = np.array([cur[int(c)] for c in common])
                q = np.array([prv[int(c)] for c in common])
                jump = _js_divergence(p, q)

        U = drift + self.beta * jump
        self.history.append(U)

        # adaptive threshold via EWMA mean/var
        fired = False
        if self.n_seen >= self.warmup:
            tau = self.ewma_mean + self.kappa * np.sqrt(self.ewma_var)
            fired = U > tau
        # update EWMA stats
        prev_mean = self.ewma_mean
        self.ewma_mean = (1 - self.rho) * self.ewma_mean + self.rho * U
        self.ewma_var = (1 - self.rho) * self.ewma_var + self.rho * (U - prev_mean) ** 2
        self.n_seen += 1

        # store for next step
        self._prev_centroids = {int(c): mu.copy()
                                for c, mu in zip(cluster_ids, centroids)}
        self._prev_mass = np.asarray(attn_mass, dtype=np.float64).copy()
        self._prev_ids = np.asarray(cluster_ids).copy()
        return U, bool(fired)