"""Load common public datasets and convert to episodic KV streams.

Supported (HuggingFace ``datasets``):
  * wikitext      — WikiText-103 paragraphs (narrative segments)
  * squad         — SQuAD v1.1 context paragraphs (RAG-style)
  * narrativeqa   — NarrativeQA story summaries (long narrative)
  * multi_news    — Multi-News articles (long-document summarization)

Each sample is tokenized (whitespace), embedded with a fixed hash projection
(reproducible, no GPU), and split into episodes by paragraph / document section.
"""
from __future__ import annotations
import os
import re
from pathlib import Path
from typing import Any

import numpy as np

_DEFAULT_HF_ROOT = os.environ.get("EPISODIC_HF_ROOT", "/root/autodl-tmp/hf")
_DEFAULT_HF_DATASETS = os.environ.get(
    "EPISODIC_HF_DATASETS_CACHE", f"{_DEFAULT_HF_ROOT}/datasets")
_DEFAULT_HF_HUB = os.environ.get(
    "EPISODIC_HF_HUB_CACHE", f"{_DEFAULT_HF_ROOT}/hub")


def _ensure_hf_env() -> None:
    """Configure mirror and on-disk caches before importing HF libraries."""
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HOME", _DEFAULT_HF_ROOT)
    os.environ.setdefault("HF_HUB_CACHE", _DEFAULT_HF_HUB)
    os.environ.setdefault("HF_DATASETS_CACHE", _DEFAULT_HF_DATASETS)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", _DEFAULT_HF_HUB)
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    for key in ("HF_HOME", "HF_HUB_CACHE", "HF_DATASETS_CACHE"):
        Path(os.environ[key]).mkdir(parents=True, exist_ok=True)


_ensure_hf_env()

# lazy registry
_REGISTRY: dict[str, Any] = {}


def _ensure_datasets():
    _ensure_hf_env()
    try:
        import datasets
        return datasets
    except ImportError as e:
        raise ImportError(
            "Install datasets: pip install datasets"
        ) from e


def _hf_load_dataset(*args, **kwargs):
    ds = _ensure_datasets()
    kwargs.setdefault("cache_dir", os.environ["HF_DATASETS_CACHE"])
    return ds.load_dataset(*args, **kwargs)


def _hash_proj(dim: int, seed: int) -> np.ndarray:
    """(vocab_bins, dim) projection table for feature hashing."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((65536, dim)) / np.sqrt(dim)


def _embed_tokens(tokens: list[str], dim: int, proj: np.ndarray,
                  episode_anchors: np.ndarray, ep_id: int,
                  anchor_w: float = 0.15) -> tuple[np.ndarray, np.ndarray]:
    """Keys = hash_embed(token) + weak episode anchor; values pack metadata."""
    keys, vals = [], []
    anchor = episode_anchors[ep_id % len(episode_anchors)]
    for tok in tokens:
        idx = hash(tok) & 0xFFFF
        k = proj[idx].copy()
        k += anchor_w * anchor
        k += 0.02 * np.random.default_rng(hash(tok) & 0xFFFFFFFF).standard_normal(dim)
        v = np.zeros(dim)
        v[0] = float(ep_id)
        v[1] = float(hash(tok) & 0xFFFF)
        v[2] = float(len(tok) % 2)   # pseudo semantic polarity for mixing metric
        keys.append(k)
        vals.append(v)
    return np.asarray(keys), np.asarray(vals)


def _build_stream(episodes: list[list[str]], dim: int = 48, seed: int = 0,
                  max_tokens_per_episode: int = 200) -> dict[str, Any]:
    proj = _hash_proj(dim, seed)
    n_ep = len(episodes)
    half = dim // 2
    rng = np.random.default_rng(seed)
    anchors = rng.standard_normal((max(n_ep, 1), dim))
    anchors /= np.linalg.norm(anchors, axis=1, keepdims=True) + 1e-8

    all_k, all_v, pos, eids, sem = [], [], [], [], []
    p = 0
    for e, toks in enumerate(episodes):
        toks = toks[:max_tokens_per_episode]
        if not toks:
            continue
        K, V = _embed_tokens(toks, dim, proj, anchors, e)
        n = len(toks)
        all_k.append(K); all_v.append(V)
        pos.extend(range(p, p + n))
        eids.extend([e] * n)
        sem.extend([e % 2] * n)   # alternating pseudo-polarity per episode
        p += n

    if not all_k:
        raise ValueError("empty episode stream")

    keys = np.vstack(all_k)
    vals = np.vstack(all_v)
    return {
        "keys": keys,
        "values": vals,
        "positions": np.asarray(pos, dtype=np.int64),
        "episode_ids": np.asarray(eids, dtype=np.int64),
        "sem_label": np.asarray(sem, dtype=np.int64),
        "dim": dim,
        "n_episodes": n_ep,
        "tokens_per_episode": max_tokens_per_episode,
        "temp": 0.25,
    }


def _tokenize(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text.strip())
    if not text:
        return []
    return text.split()


def _split_paragraphs(text: str, min_words: int = 20) -> list[list[str]]:
    paras = re.split(r"\n\s*\n+|\n=+ .+ =+\n", text)
    out = []
    for p in paras:
        toks = _tokenize(p)
        if len(toks) >= min_words:
            out.append(toks)
    return out


# ------------------------------------------------------------------ loaders
def load_wikitext(n_docs: int = 5, dim: int = 48, seed: int = 0,
                  split: str = "test") -> dict[str, Any]:
    ds = _hf_load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split=split)
    episodes: list[list[str]] = []
    for row in ds:
        if row.get("text", "").strip().startswith("="):
            continue
        paras = _split_paragraphs(row["text"], min_words=15)
        episodes.extend(paras)
        if len(episodes) >= n_docs * 8:
            break
    episodes = episodes[: n_docs * 8]
    stream = _build_stream(episodes, dim=dim, seed=seed)
    stream["source"] = "wikitext-103"
    stream["meta"] = {"n_docs": n_docs, "split": split}
    return stream


def load_squad(n_contexts: int = 10, dim: int = 48, seed: int = 0,
               split: str = "validation") -> dict[str, Any]:
    ds = _hf_load_dataset("rajpurkar/squad", split=split)
    episodes = []
    for row in ds:
        ctx = row.get("context", "")
        toks = _tokenize(ctx)
        if len(toks) < 30:
            continue
        # split long contexts into ~100-word episodes
        chunk = 100
        for i in range(0, len(toks), chunk):
            episodes.append(toks[i:i + chunk])
        if len(episodes) >= n_contexts * 3:
            break
    episodes = episodes[: n_contexts * 3]
    stream = _build_stream(episodes, dim=dim, seed=seed)
    stream["source"] = "squad-v1.1"
    stream["meta"] = {"n_contexts": n_contexts, "split": split}
    return stream


def load_narrativeqa(n_stories: int = 5, dim: int = 48, seed: int = 0,
                     split: str = "validation") -> dict[str, Any]:
    try:
        ds = _hf_load_dataset("narrativeqa", split=split)
        source = "narrativeqa"
    except Exception:
        # fallback: use a small narrative-style subset from wikitext
        ds = _hf_load_dataset(
            "Salesforce/wikitext", "wikitext-103-raw-v1", split="test")
        source = "wikitext_fallback"
    episodes = []
    for row in ds:
        if "summary" in row:
            summary = row.get("summary", {}).get("text", "") if isinstance(
                row.get("summary"), dict) else str(row.get("summary", ""))
        else:
            summary = row.get("text", "")
        if not summary:
            continue
        paras = _split_paragraphs(summary, min_words=10)
        if not paras:
            toks = _tokenize(summary)
            if len(toks) >= 20:
                episodes.append(toks)
        else:
            episodes.extend(paras)
        if len(episodes) >= n_stories * 6:
            break
    episodes = episodes[: n_stories * 6]
    stream = _build_stream(episodes, dim=dim, seed=seed)
    stream["source"] = source
    stream["meta"] = {"n_stories": n_stories, "split": split}
    return stream


def load_multi_news(n_articles: int = 5, dim: int = 48, seed: int = 0,
                    split: str = "validation") -> dict[str, Any]:
    try:
        ds = _hf_load_dataset("multi_news", split=split)
        source = "multi_news"
        field = "document"
    except Exception:
        ds = _hf_load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test")
        source = "wikitext_long_fallback"
        field = "text"
    episodes = []
    for row in ds:
        doc = row.get(field, "")
        paras = _split_paragraphs(doc.replace("|||||", "\n\n"), min_words=15)
        episodes.extend(paras[:8])
        if len(episodes) >= n_articles * 8:
            break
    episodes = episodes[: n_articles * 8]
    stream = _build_stream(episodes, dim=dim, seed=seed)
    stream["source"] = source
    stream["meta"] = {"n_articles": n_articles, "split": split}
    return stream


DATASET_LOADERS = {
    "wikitext": load_wikitext,
    "squad": load_squad,
    "narrativeqa": load_narrativeqa,
    "multi_news": load_multi_news,
}

# smoke = few docs; full = default loader kwargs
DATASET_PRESETS = {
    "smoke": {
        "wikitext": {"n_docs": 1},
        "squad": {"n_contexts": 2},
        "narrativeqa": {"n_stories": 1},
        "multi_news": {"n_articles": 1},
    },
    "full": {
        "wikitext": {"n_docs": 5},
        "squad": {"n_contexts": 10},
        "narrativeqa": {"n_stories": 5},
        "multi_news": {"n_articles": 5},
    },
}


def load_dataset(name: str, preset: str | None = None, **kwargs) -> dict[str, Any]:
    """Load a registered public dataset by short name.

    ``preset`` may be ``smoke`` (few examples) or ``full`` (default scale).
    Explicit ``kwargs`` override preset values.
    """
    if name not in DATASET_LOADERS:
        raise KeyError(f"unknown dataset {name!r}; choose from {list(DATASET_LOADERS)}")
    kw = dict(DATASET_PRESETS.get(preset or "full", {}).get(name, {}))
    kw.update(kwargs)
    return DATASET_LOADERS[name](**kw)


def list_datasets() -> list[str]:
    return list(DATASET_LOADERS)
