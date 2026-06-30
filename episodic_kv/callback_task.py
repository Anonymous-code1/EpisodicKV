"""A clean, decisive 'callback' task that isolates the episodic-conflict problem.

Construction:
  * A document of E episodes. One designated entity (the 'subject') appears in
    SEVERAL episodes; its polarity FLIPS between episodes (conflict).
  * The decoding head sits at the end. The query is a CALLBACK: "what was the
    subject's state during episode e*?", where e* is a mid-document episode whose
    subject tokens are (a) OUTSIDE the local window and (b) surrounded by other
    episodes carrying the OPPOSITE polarity for the same subject.
  * Answer = the subject's polarity in episode e*. To get it right, the cache
    must retrieve the subject's tokens FROM EPISODE e* specifically, not the
    nearby conflicting ones.

This is the regime where pure key-similarity retrieval fails (all the subject's
tokens look alike) and Episode-Exclusive Retrieval wins (it can target e*).
The 'active episode' for exclusion is e* (the episode the query refers to),
which a real model supplies via the query's positional/episodic content.
"""
from __future__ import annotations
import numpy as np


def make_callback_doc(n_episodes=12, tokens_per_episode=120, dim=48,
                      n_filler_entities=5, seed=0):
    rng = np.random.default_rng(seed)
    half = dim // 2
    episode_anchor = np.zeros((n_episodes, dim))
    ea = rng.standard_normal((n_episodes, half)); ea /= np.linalg.norm(ea, axis=1, keepdims=True)
    episode_anchor[:, :half] = ea

    # subject proto + filler protos (2nd half)
    n_ent = n_filler_entities + 1
    protos = np.zeros((n_ent, dim))
    pp = rng.standard_normal((n_ent, dim - half)); pp /= np.linalg.norm(pp, axis=1, keepdims=True)
    protos[:, half:] = pp
    SUBJECT = 0

    KEY_ANCHOR_W = 0.18   # weak episode signal in the key -> F1
    # subject polarity per episode: alternating blocks create stale decoys
    subj_pol = np.array([(e // 2) % 2 for e in range(n_episodes)])

    keys, vals, eid, ent_id, pos = [], [], [], [], []
    p = 0
    for e in range(n_episodes):
        anchor = episode_anchor[e]
        for _ in range(tokens_per_episode):
            # ~35% of tokens are the subject, rest filler
            if rng.random() < 0.35:
                ent = SUBJECT; sem = int(subj_pol[e])
            else:
                ent = int(rng.integers(1, n_ent)); sem = int(rng.integers(0, 2))
            proto = protos[ent]
            k = proto + KEY_ANCHOR_W * anchor + 0.05 * rng.standard_normal(dim)
            v = np.zeros(dim); v[0] = e; v[1] = ent; v[2] = float(sem)
            keys.append(k); vals.append(v); eid.append(e); ent_id.append(ent); pos.append(p)
            p += 1
    return {
        "keys": np.array(keys), "values": np.array(vals),
        "episode_ids": np.array(eid), "entity_ids": np.array(ent_id),
        "positions": np.array(pos), "episode_anchor": episode_anchor,
        "protos": protos, "subject": SUBJECT, "subj_pol": subj_pol,
        "dim": dim, "n_episodes": n_episodes, "tokens_per_episode": tokens_per_episode,
        "temp": 0.25, "key_anchor_w": KEY_ANCHOR_W,
    }


def render_callback_conflict_context(doc, target_episode: int, *, distractor_span: int = 2) -> str:
    parts: list[str] = []
    n_episodes = int(doc["n_episodes"])
    subj_pol = doc["subj_pol"]
    subject_name = "subject-omega"
    for e in range(n_episodes):
        state = "yes" if int(subj_pol[e]) == 1 else "no"
        stale_state = "no" if state == "yes" else "yes"
        lines = [
            f"Episode {e} record for {subject_name}.",
            f"Canonical state for this episode: {state}.",
            f"This episode-specific state overrides earlier episodes for the same subject.",
            f"Do not answer using adjacent episodes when a direct episode reference is provided.",
        ]
        if abs(e - target_episode) <= distractor_span and e != target_episode:
            lines.extend([
                f"Distractor note: nearby episode {e} reports {stale_state} in a conflicting recap.",
                f"That recap should not be used for episode {target_episode}.",
            ])
        if e == target_episode:
            lines.extend([
                f"Target callback anchor: the requested episode is exactly Episode {target_episode}.",
                f"Any later summary claiming {stale_state} belongs to a different episode.",
            ])
        lines.extend([
            f"Archive tag {e}: {subject_name} state {state}.",
            f"Checksum phrase episode-{e}-state-{state}.",
        ])
        parts.append(" ".join(lines))
    tail_distractors = []
    for k in range(max(3, distractor_span + 1)):
        wrong_episode = (target_episode + k + 1) % n_episodes
        wrong_state = "yes" if int(subj_pol[wrong_episode]) == 1 else "no"
        tail_distractors.append(
            f"Late recap {k}: episode {wrong_episode} says {wrong_state}; this should not replace episode {target_episode}."
        )
    parts.append(" ".join(tail_distractors))
    return "\n\n".join(parts)


def callback_query(doc, target_episode):
    """Query asking for the SUBJECT's state during `target_episode`.
    Returns q, relevant_mask (subject tokens in target_episode), answer."""
    eid, ent = doc["episode_ids"], doc["entity_ids"]
    anchor = doc["episode_anchor"][target_episode]
    proto = doc["protos"][doc["subject"]]
    # query addresses the TARGET episode (strong) + the subject
    q = 2.0 * anchor + proto
    rel = (eid == target_episode) & (ent == doc["subject"])
    ans = int(doc["subj_pol"][target_episode])
    return q, rel, ans