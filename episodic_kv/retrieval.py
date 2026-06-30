"""Episode-Exclusive Retrieval (EER) + accuracy-bounded (Jensen) estimation.

This is the core algorithmic contribution (Sec. 2.3 of PLAN.md).

Two-level use of the episode coordinate:

  (1) Cluster selection (which clusters to fetch precisely):
        s_c = q.mu_c - gamma * U_c + eta * recency_c
  (2) Episode-EXCLUSIVE per-token weighting (THE key step). Standard sparse KV
      caches compute exact softmax over retrieved member tokens, so stale tokens
      whose keys are nearly identical to current ones are re-admitted at full
      weight. EER instead adds an *episode-compatibility bias* to each token's
      logit:
            logit_i  <-  q.k_i / temp  +  log pi(c(i), t)
            pi(c, t) = exp( -rho * d_episode(c, active_episode_t) )
      so tokens from stale/conflicting episodes are suppressed EXPONENTIALLY.
      A plain k-means index has no episode coordinate => pi == 1 => no exclusion
      => it cannot avoid stale tokens. This is the mechanism that separates the
      methods, and the basis of the tight-error-bound theorem.
"""
from __future__ import annotations
import numpy as np


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def episode_decayed_attention(q: np.ndarray,
                              index,
                              steady_keys: np.ndarray,
                              steady_vals: np.ndarray,
                              r_clusters: int,
                              gamma: float = 0.5,
                              eta: float = 0.1,
                              conflict_per_cluster: np.ndarray | None = None,
                              d: int | None = None,
                              temp: float | None = None,
                              rho: float = 0.0,
                              active_episode_coord: np.ndarray | None = None,
                              active_episode_id: int | None = None,
                              return_diagnostics: bool = False):
    """Compute approximate attention output for query q.

    rho > 0 enables Episode-Exclusive Retrieval. `active_episode_coord` is the
    episode-field coordinate of the current (active) episode; clusters far from
    it in episode space are exponentially down-weighted at the token level.
    For a plain k-means index (no episode coords), pass rho=0 -> standard
    precise retrieval (no exclusion).
    """
    d = d or q.shape[-1]
    scale = 1.0 / (temp if temp is not None else np.sqrt(d))
    C = index.centroids()                  # (m, d)
    S = index.sizes()                      # (m,)
    VS = index.value_sums()                # (m, d)
    rec = index.recency()                  # (m,)
    m = C.shape[0]

    if conflict_per_cluster is None:
        conflict_per_cluster = np.zeros(m)

    # Episode-Exclusive: score_final = score_cluster * exp(-λ * episode_gap)
    episode_decay = np.ones(m)
    log_pi = np.zeros(m)
    if rho > 0.0:
        if active_episode_coord is not None \
                and hasattr(index, "cluster_episode_coord"):
            cc = index.cluster_episode_coord()
            dist = np.linalg.norm(cc - active_episode_coord[None, :], axis=1)
            episode_decay = np.exp(-rho * dist)
            log_pi = np.log(episode_decay + 1e-12)
        elif hasattr(index, "episodes"):
            eps = index.episodes()
            active_ep = active_episode_id if active_episode_id is not None \
                else (int(eps.max()) if eps.size else 0)
            gap = np.abs(eps.astype(np.float64) - active_ep)
            episode_decay = np.exp(-rho * gap)
            log_pi = np.log(episode_decay + 1e-12)

    # --- cluster selection: episode-decayed score (+ compatibility) ---
    centroid_scores = (C @ q) * scale                      # q . mu_c / temp
    rec_norm = np.log1p(rec - rec.min() + 1.0)
    rec_norm = rec_norm / (rec_norm.max() + 1e-8)
    sel_score = (centroid_scores * episode_decay - gamma * conflict_per_cluster
                 + eta * rec_norm)
    r_clusters = min(r_clusters, m)
    retr = np.argsort(-sel_score)[:r_clusters]
    est_mask = np.ones(m, dtype=bool)
    est_mask[retr] = False

    steady_logits = (steady_keys @ q) * scale if steady_keys.shape[0] else np.zeros(0)
    cluster_logits = centroid_scores
    big = max(steady_logits.max() if steady_logits.size else -1e30,
              cluster_logits.max() if cluster_logits.size else -1e30)
    steady_exp = np.exp(steady_logits - big)
    cluster_exp = np.exp(cluster_logits + log_pi - big)   # bias estimation too

    o = np.zeros(d)
    if steady_keys.shape[0]:
        o += (steady_exp[:, None] * steady_vals).sum(axis=0)

    have_members = hasattr(index, "mem_keys") and len(getattr(index, "mem_keys")) == m
    if have_members:
        denom = steady_exp.sum() + 1e-30
        denom += (S[est_mask] * cluster_exp[est_mask]).sum()
        for c in retr:
            mk = index.mem_keys[c]
            mv = index.mem_vals[c]
            # EPISODE-EXCLUSIVE per-token logit: add cluster's compatibility bias
            lg = (mk @ q) * scale + log_pi[c]
            ex = np.exp(lg - big)
            o += (ex[:, None] * mv).sum(axis=0)
            denom += ex.sum()
        o += (cluster_exp[est_mask][:, None] * VS[est_mask]).sum(axis=0)
    else:
        denom = steady_exp.sum() + (S * cluster_exp).sum() + 1e-30
        for c in retr:
            o += cluster_exp[c] * VS[c]
        o += (cluster_exp[est_mask][:, None] * VS[est_mask]).sum(axis=0)

    o = o / denom

    if return_diagnostics:
        mass = (S * cluster_exp) / denom
        diag = {
            "cluster_logits": cluster_logits,
            "sel_score": sel_score,
            "retrieval_clusters": retr,
            "attn_mass": mass,
            "log_pi": log_pi,
            "denom": denom,
        }
        return o, diag
    return o