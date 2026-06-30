"""Synthetic episodic stream with a CLEAN, well-posed conflict task.

Design goals (validated by `oracle_check`):
  * Full attention is a strong oracle (>0.8 mass on the correct tokens).
  * The ONLY way to get the answer right is to read the entity's tokens from the
    CURRENT episode, not from earlier episodes where its meaning conflicts.
  * Key similarity alone cannot separate current vs stale tokens of an entity
    (same surface proto) -> pure-similarity clustering fails (F1). Temporal phase
    (episode field) can.

Task: at query time we ask about entity `tgt` "as of now" (current episode).
The answer is its current-episode polarity in {0,1}. Earlier episodes may carry
the opposite polarity (conflict). The value vector packs the polarity so we can
read it from the attention output.
"""
from __future__ import annotations
import numpy as np


def make_episodic_stream(
    n_episodes: int = 12,
    tokens_per_episode: int = 200,
    dim: int = 64,
    n_entities: int = 6,
    conflict_rate: float = 0.5,
    drift: float = 0.0,
    temp: float = 0.25,          # softmax temperature used by the oracle
    seed: int = 0,
):
    rng = np.random.default_rng(seed)

    # Episode anchors and entity protos. CRUCIAL design (the F1 failure mode):
    # an entity's KEY is dominated by its proto and carries only a WEAK episode
    # signal, so current vs stale tokens of the SAME entity are nearly identical
    # in key space. Pure key-similarity clustering therefore pools all episodes
    # of an entity together and retrieval cannot avoid stale (conflicting)
    # tokens. The episode field adds an explicit temporal phase that recovers the
    # weak episode signal and separates them.
    half = dim // 2
    episode_anchor = np.zeros((n_episodes, dim))
    ea = rng.standard_normal((n_episodes, half))
    ea /= np.linalg.norm(ea, axis=1, keepdims=True)
    episode_anchor[:, :half] = ea                      # anchors in 1st half

    entity_proto = np.zeros((n_entities, dim))
    ep = rng.standard_normal((n_entities, dim - half))
    ep /= np.linalg.norm(ep, axis=1, keepdims=True)
    entity_proto[:, half:] = ep                        # protos in 2nd half

    # weight of the episode signal *inside the key* (small => keys of an entity
    # look the same across episodes => k-means cannot separate => F1).
    KEY_ANCHOR_W = 0.18

    # per-(episode, entity) polarity, with episodic conflict on reappearance
    first_polarity = {}
    ep_ent_polarity = np.zeros((n_episodes, n_entities), dtype=int)
    for e in range(n_episodes):
        for en in range(n_entities):
            if en not in first_polarity:
                first_polarity[en] = int(rng.integers(0, 2))
                ep_ent_polarity[e, en] = first_polarity[en]
            else:
                flip = rng.random() < conflict_rate
                ep_ent_polarity[e, en] = (1 - first_polarity[en]) if flip \
                    else first_polarity[en]

    keys, vals, pos, eid, ent_id, sem_label = [], [], [], [], [], []
    p = 0
    for e in range(n_episodes):
        anchor = episode_anchor[e].copy()
        if drift:
            anchor += drift * episode_anchor[(e - 1) % n_episodes]
            anchor /= np.linalg.norm(anchor)
        for _ in range(tokens_per_episode):
            ent = int(rng.integers(0, n_entities))
            proto = entity_proto[ent]
            # KEY = proto (strong entity addr) + WEAK episode anchor + noise.
            # The weak anchor is below the resolution of static k-means but is
            # exactly what the episode-field temporal phase reconstructs.
            k = proto + KEY_ANCHOR_W * anchor + 0.05 * rng.standard_normal(dim)
            sem = int(ep_ent_polarity[e, ent])
            v = np.zeros(dim)
            v[0] = e
            v[1] = ent
            v[2] = float(sem)
            keys.append(k)
            vals.append(v)
            pos.append(p)
            eid.append(e)
            ent_id.append(ent)
            sem_label.append(sem)
            p += 1

    return {
        "keys": np.array(keys), "values": np.array(vals),
        "positions": np.array(pos), "episode_ids": np.array(eid),
        "entity_ids": np.array(ent_id), "sem_label": np.array(sem_label),
        "episode_anchor": episode_anchor, "entity_proto": entity_proto,
        "dim": dim, "n_episodes": n_episodes, "n_entities": n_entities,
        "tokens_per_episode": tokens_per_episode, "temp": temp,
    }


def make_current_episode_query(stream, current_pos, dim, rng):
    """Query about entity `tgt` AS OF the current episode. The relevant tokens
    are tgt's tokens in the CURRENT episode; reading earlier (stale) episodes of
    the same entity may flip the answer (conflict).

    The query carries a STRONG current-episode anchor (the model "knows" which
    episode it is in) plus the entity proto. The difficulty is on the KV side:
    the cached keys of the entity barely differ across episodes, so retrieval
    must use temporal structure to pick the current-episode tokens.
    """
    eid, ent = stream["episode_ids"], stream["entity_ids"]
    cur_e = int(eid[current_pos])
    seen = np.arange(current_pos + 1)
    cur_ents = np.unique(ent[seen][eid[seen] == cur_e])
    # prefer an entity that ALSO appeared earlier with a different polarity
    # (a genuine conflict), to make the test discriminative
    tgt = None
    rng.shuffle(cur_ents)
    for cand in cur_ents:
        cur_pol = stream["sem_label"][seen][(eid[seen] == cur_e) & (ent[seen] == cand)]
        old_pol = stream["sem_label"][seen][(eid[seen] != cur_e) & (ent[seen] == cand)]
        if cur_pol.size and old_pol.size and old_pol.mean() != cur_pol.mean():
            tgt = int(cand); break
    if tgt is None:
        tgt = int(cur_ents[0]) if cur_ents.size else 0
    anchor = stream["episode_anchor"][cur_e]
    proto = stream["entity_proto"][tgt]
    q = 2.0 * anchor + proto                  # strong episode address + entity
    rel = seen[(eid[seen] == cur_e) & (ent[seen] == tgt)]
    mask = np.zeros(current_pos + 1, dtype=bool)
    mask[rel] = True
    return q, mask, cur_e, tgt


def oracle_check(seed=0, n_q=300):
    """Sanity: full attention should put most mass on relevant tokens and answer
    correctly. Returns (mass_on_relevant, oracle_accuracy)."""
    s = make_episodic_stream(n_episodes=8, tokens_per_episode=120, dim=48,
                             conflict_rate=0.6, seed=seed)
    keys, vals = s["keys"], s["values"]
    dim, temp, n = s["dim"], s["temp"], len(keys)
    rng = np.random.default_rng(seed + 1)
    masses, accs = [], []
    for _ in range(n_q):
        cp = int(rng.integers(n // 2, n))
        q, mask, cur_e, tgt = make_current_episode_query(s, cp, dim, rng)
        if not mask.any():
            continue
        logits = (keys[:cp + 1] @ q) / temp
        w = np.exp(logits - logits.max()); w /= w.sum()
        o = w @ vals[:cp + 1]
        masses.append(w[mask].sum())
        true = int(round(vals[:cp + 1][mask][:, 2].mean()))
        accs.append((1 if o[2] > 0.5 else 0) == true)
    return float(np.mean(masses)), float(np.mean(accs))


if __name__ == "__main__":
    m, a = oracle_check()
    print(f"oracle mass on relevant = {m:.3f}   oracle accuracy = {a:.3f}")