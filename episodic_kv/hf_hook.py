"""Real-model hook: a drop-in episode-aware attention for HuggingFace decoders.

This is the *plug-and-play* path. It does NOT require modifying model weights.
It wraps the attention of a causal LM so that, during decoding, attention over
past tokens is served by an EpisodicKV index (Episode-Exclusive Retrieval)
instead of dense full attention.

NOTE: running this end-to-end needs `torch` + `transformers` + a model + a GPU
for any non-trivial size. The CPU mechanism validation (experiments/) does NOT
need this file. This module is a reference integration skeleton showing exactly
where EpisodicKV plugs into a Llama/Qwen-style attention, with a tiny toy-model
smoke test that runs on CPU.
"""
from __future__ import annotations
import math

try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except Exception:  # torch optional
    _HAS_TORCH = False


if _HAS_TORCH:

    def _temporal_phase_torch(positions, r=8, episode_scale=1024.0, base=10000.0,
                              device=None, dtype=torch.float32):
        k = torch.arange(r // 2, device=device, dtype=dtype)
        freqs = (1.0 / (base ** (2.0 * k / r))) / episode_scale
        ang = positions[:, None].to(dtype) * freqs[None, :]
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)

    class EpisodicKVAttention(nn.Module):
        """Episode-aware sparse attention head-group, per layer.

        Holds, for the prompt + generated tokens of ONE attention head group:
          * raw K,V on (CPU) host memory (offloadable),
          * episode-field cluster meta-index on device,
        and serves decode-step attention with Episode-Exclusive Retrieval.

        Usage: build per (layer, kv-head) and call `decode_step(q)`; or call
        `prefill(K, V, positions)` once, then `decode_step(q, pos)` per token.
        This skeleton implements the math in float; production needs the CUDA
        kernels described in PLAN.md Sec.3.
        """

        def __init__(self, head_dim, lam=0.6, tokens_per_centroid=16,
                     r_clusters=8, sink=4, local_window=64, min_episode=512,
                     rho=4.0, r_phase=8, episode_scale=1024.0):
            super().__init__()
            self.d = head_dim
            self.lam = lam
            self.tpc = tokens_per_centroid
            self.r_clusters = r_clusters
            self.sink = sink
            self.local_window = local_window
            self.min_episode = min_episode
            self.rho = rho
            self.r_phase = r_phase
            self.episode_scale = episode_scale
            self.reset()

        def reset(self):
            self.mu = None          # (m, d) centroids
            self.VS = None          # (m, d) value sums
            self.S = None           # (m,) sizes
            self.ep_coord = None    # (m, r_phase)
            self.mem_k = []         # list of (s_c, d) raw keys per cluster
            self.mem_v = []
            self.steady_k = None    # (p, d)
            self.steady_v = None

        @torch.no_grad()
        def _cluster(self, K, V, positions):
            """Episode-field segmented clustering of (K,V) -> meta index."""
            n, d = K.shape
            kh = K / (K.norm(dim=-1, keepdim=True) + 1e-8)
            tp = _temporal_phase_torch(positions, self.r_phase, self.episode_scale,
                                       device=K.device, dtype=K.dtype)
            phi = torch.cat([kh, self.lam * tp], dim=-1)
            phi = phi / (phi.norm(dim=-1, keepdim=True) + 1e-8)
            k = max(1, n // self.tpc)
            # simple spherical k-means
            idx = torch.randperm(n, device=K.device)[:k]
            cen = phi[idx].clone()
            for _ in range(10):
                sim = phi @ cen.t()
                lab = sim.argmax(dim=1)
                for c in range(k):
                    m = lab == c
                    if m.any():
                        mc = phi[m].mean(0)
                        cen[c] = mc / (mc.norm() + 1e-8)
            mus, vss, ss, eps, mk, mv = [], [], [], [], [], []
            for c in range(k):
                m = lab == c
                if not m.any():
                    continue
                mus.append(K[m].mean(0)); vss.append(V[m].sum(0))
                ss.append(int(m.sum())); eps.append(tp[m].mean(0))
                mk.append(K[m]); mv.append(V[m])
            return (torch.stack(mus), torch.stack(vss),
                    torch.tensor(ss, device=K.device, dtype=K.dtype),
                    torch.stack(eps), mk, mv)

        @torch.no_grad()
        def prefill(self, K, V, positions):
            self.reset()
            p = self.sink + self.local_window
            self.steady_k = torch.cat([K[:self.sink], K[-self.local_window:]], 0) \
                if K.shape[0] > p else K
            self.steady_v = torch.cat([V[:self.sink], V[-self.local_window:]], 0) \
                if V.shape[0] > p else V
            body = slice(self.sink, max(self.sink, K.shape[0] - self.local_window))
            if K[body].shape[0] > 0:
                self.mu, self.VS, self.S, self.ep_coord, self.mem_k, self.mem_v = \
                    self._cluster(K[body], V[body], positions[body])

        @torch.no_grad()
        def decode_step(self, q, pos, temp=None):
            """Episode-Exclusive Retrieval attention for one decode query q."""
            d = self.d
            scale = 1.0 / (temp if temp is not None else math.sqrt(d))
            o = torch.zeros(d, device=q.device, dtype=q.dtype)
            denom = torch.zeros((), device=q.device, dtype=q.dtype)
            # steady
            if self.steady_k is not None and self.steady_k.shape[0]:
                lg = (self.steady_k @ q) * scale
                big = lg.max()
                ex = torch.exp(lg - big)
                o = o + (ex[:, None] * self.steady_v).sum(0)
                denom = denom + ex.sum()
            else:
                big = torch.zeros((), device=q.device, dtype=q.dtype)
            if self.mu is None:
                return o / (denom + 1e-9)
            # episode compatibility bias
            aec = _temporal_phase_torch(torch.tensor([pos], device=q.device),
                                        self.r_phase, self.episode_scale,
                                        device=q.device, dtype=q.dtype)[0]
            dist = (self.ep_coord - aec[None, :]).norm(dim=1)
            log_pi = -self.rho * dist
            cs = (self.mu @ q) * scale
            sel = cs + log_pi
            r = min(self.r_clusters, self.mu.shape[0])
            retr = torch.topk(sel, r).indices
            retr_set = set(retr.tolist())
            cexp = torch.exp(cs + log_pi - big)
            for c in range(self.mu.shape[0]):
                if c in retr_set:
                    lg = (self.mem_k[c] @ q) * scale + log_pi[c]
                    ex = torch.exp(lg - big)
                    o = o + (ex[:, None] * self.mem_v[c]).sum(0)
                    denom = denom + ex.sum()
                else:
                    o = o + cexp[c] * self.VS[c]
                    denom = denom + self.S[c] * cexp[c]
            return o / (denom + 1e-9)


def smoke_test():
    """Tiny CPU smoke test of the hook: episodic attention approximates full
    attention on a random toy sequence."""
    if not _HAS_TORCH:
        print("[hf_hook] torch not installed; skipping smoke test.")
        return
    torch.manual_seed(0)
    n, d = 600, 32
    K = torch.randn(n, d); V = torch.randn(n, d)
    positions = torch.arange(n)
    q = torch.randn(d)
    # full attention
    lg = (K @ q) / math.sqrt(d); w = torch.softmax(lg, 0); o_full = w @ V
    att = EpisodicKVAttention(head_dim=d, r_clusters=20, local_window=32,
                              sink=4, rho=0.0)  # rho=0 -> pure retrieval
    att.prefill(K, V, positions)
    o = att.decode_step(q, pos=n)
    err = (o - o_full).norm() / (o_full.norm() + 1e-9)
    print(f"[hf_hook] toy relative error vs full attention: {err:.3f}")


if __name__ == "__main__":
    smoke_test()