from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import math
import re
import uuid
import time
import os

from .paged_pool import GlobalPrefixPool, PageAllocator
from .cache import EpisodicKVCache


DEFAULT_MODEL_PATH = os.environ.get(
    "EPISODIC_MODEL_PATH",
    "/root/autodl-tmp/models/Qwen2.5-7B-Instruct",
)


@dataclass
class GenerationSession:
    session_id: str
    prompt: str
    prompt_tokens: int
    method: str
    dataset: str | None = None
    generated_text: str = ""
    generated_tokens: int = 0
    prefix_hit: bool = False
    prefix_key: str | None = None
    page_ids: list[int] = field(default_factory=list)
    episode_boundaries: list[int] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    steady_words: int = 0
    retrieval_words: int = 0
    estimation_words: int = 0


class EpisodicRuntime:
    """Minimal industrial runtime for Qwen2.5-7B experiments.

    This runtime intentionally separates:
    - shared prefix/page accounting,
    - session-local KV/page accounting,
    - method switching for baseline vs episodic behavior.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        device_map: str = "auto",
        torch_dtype: str = "float16",
        page_size_tokens: int = 128,
        method: str = "full",
        max_input_tokens: int = 4096,
        episodic_config: dict[str, Any] | None = None,
    ):
        self.model_path = model_path
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.page_allocator = PageAllocator(page_size_tokens=page_size_tokens)
        self.prefix_pool = GlobalPrefixPool(self.page_allocator)
        self.method = method
        self.max_input_tokens = max_input_tokens
        self.episodic_config = episodic_config or {}
        self.tokenizer = None
        self.model = None
        self.sessions: dict[str, GenerationSession] = {}
        self._loaded = False

    def load(self):
        if self._loaded:
            return self
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        dtype = getattr(torch, self.torch_dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        cfg = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)

        base_ctx = int(getattr(cfg, "max_position_embeddings", 0) or 0)
        tok_ctx = int(getattr(self.tokenizer, "model_max_length", 0) or 0)
        desired_ctx = int(max(self.max_input_tokens, tok_ctx, base_ctx))
        if base_ctx > 0 and desired_ctx > base_ctx:
            factor = max(float(desired_ctx) / float(base_ctx), 1.0)
            rope_theta = float(getattr(cfg, "rope_theta", 1000000.0) or 1000000.0)
            cfg.rope_parameters = {
                "rope_type": "yarn",
                "rope_theta": rope_theta,
                "factor": factor,
                "original_max_position_embeddings": base_ctx,
            }
            cfg.rope_scaling = {
                "rope_type": "yarn",
                "rope_theta": rope_theta,
                "factor": factor,
                "original_max_position_embeddings": base_ctx,
            }
            cfg.max_position_embeddings = desired_ctx
            if getattr(cfg, "sliding_window", None):
                cfg.sliding_window = max(int(cfg.sliding_window), desired_ctx)

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            config=cfg,
            dtype=dtype,
            device_map=self.device_map,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self._loaded = True
        return self

    def create_session(
        self,
        prompt: str,
        *,
        method: str | None = None,
        dataset: str | None = None,
        session_id: str | None = None,
    ) -> GenerationSession:
        self.load()
        sid = session_id or f"sess-{uuid.uuid4().hex[:8]}"
        toks = self.tokenizer(prompt, return_tensors="pt", truncation=True,
                              max_length=self.max_input_tokens)
        prompt_tokens = int(toks["input_ids"].shape[1])
        sess = GenerationSession(
            session_id=sid,
            prompt=prompt,
            prompt_tokens=prompt_tokens,
            method=method or self.method,
            dataset=dataset,
        )
        prefix = self.prefix_pool.lookup(prompt)
        if prefix is not None:
            sess.prefix_hit = True
            sess.prefix_key = prefix["key"]
        else:
            prefix = self.prefix_pool.register(prompt, prompt_tokens)
            sess.prefix_key = prefix["key"]
        pages = self.page_allocator.allocate_tokens(
            sid, prompt_tokens, scope="private", meta={"kind": "prompt"}, generated=False)
        sess.page_ids.extend([p.page_id for p in pages])
        self.sessions[sid] = sess
        return sess

    def _zone_config(self, n_words: int, method: str) -> dict[str, int]:
        steady = int(self.episodic_config.get("steady_words", 192))
        chunk = int(self.episodic_config.get("chunk_words", 96))
        retrieval_ratio = float(self.episodic_config.get("retrieval_budget_ratio", 0.25))
        retrieval_min = int(self.episodic_config.get("retrieval_budget_min", 2))
        retrieval_max = int(self.episodic_config.get("retrieval_budget_max", 8))
        n_chunks = max(1, math.ceil(max(n_words, 1) / max(chunk, 1)))
        retrieval_chunks = max(retrieval_min, int(round(n_chunks * retrieval_ratio)))
        retrieval_chunks = min(retrieval_chunks, retrieval_max, n_chunks)
        if method == "sliding":
            retrieval_chunks = 0
        return {
            "steady_words": steady,
            "chunk_words": chunk,
            "retrieval_chunks": retrieval_chunks,
        }

    def _split_prompt_zones(self, prompt: str, method: str) -> dict[str, Any]:
        words = prompt.split()
        cfg = self._zone_config(len(words), method)
        steady_words = min(cfg["steady_words"], len(words))
        chunk_words = max(cfg["chunk_words"], 1)
        retrieval_chunks = cfg["retrieval_chunks"]
        if method == "full":
            return {
                "prompt": prompt,
                "steady_words": len(words),
                "retrieval_words": 0,
                "estimation_words": 0,
                "selected_chunk_ids": [],
            }
        if method == "sliding":
            kept = words[-max(steady_words * 2, 1024):]
            return {
                "prompt": " ".join(kept),
                "steady_words": len(kept),
                "retrieval_words": 0,
                "estimation_words": max(len(words) - len(kept), 0),
                "selected_chunk_ids": [],
            }

        retrieval_source = words[:-steady_words] if steady_words < len(words) else []
        steady_zone = words[-steady_words:] if steady_words > 0 else words
        chunks = [retrieval_source[i:i + chunk_words] for i in range(0, len(retrieval_source), chunk_words)]
        chunks = [c for c in chunks if c]
        selected_ids: list[int] = []
        selected_chunks: list[list[str]] = []
        if method == "fixed":
            selected_chunks = chunks[-retrieval_chunks:] if retrieval_chunks > 0 else []
            selected_ids = list(range(max(len(chunks) - len(selected_chunks), 0), len(chunks)))
        elif method == "kmeans":
            selected_chunks, selected_ids = self._select_kmeans_chunks(chunks, retrieval_chunks, steady_zone)
        elif method == "episodic":
            selected_chunks, selected_ids = self._select_episodic_chunks(chunks, retrieval_chunks, steady_zone)
        else:
            selected_chunks = chunks[-retrieval_chunks:] if retrieval_chunks > 0 else []
            selected_ids = list(range(max(len(chunks) - len(selected_chunks), 0), len(chunks)))

        selected_words = [w for c in selected_chunks for w in c]
        final_words = selected_words + steady_zone
        return {
            "prompt": " ".join(final_words) if final_words else prompt,
            "steady_words": len(steady_zone),
            "retrieval_words": len(selected_words),
            "estimation_words": max(len(words) - len(final_words), 0),
            "selected_chunk_ids": selected_ids,
        }

    def _select_kmeans_chunks(self, chunks: list[list[str]], retrieval_chunks: int,
                              steady_zone: list[str]) -> tuple[list[list[str]], list[int]]:
        if not chunks or retrieval_chunks <= 0:
            return [], []
        import numpy as np
        from .datasets import _embed_tokens, _hash_proj
        from .episode_field import _l2norm

        dim = int(self.episodic_config.get("dim", 48))
        seed = int(self.episodic_config.get("seed", 0))
        proj = _hash_proj(dim, seed)
        rng = np.random.default_rng(seed)
        anchors = rng.standard_normal((max(len(chunks), 1), dim))
        anchors /= np.linalg.norm(anchors, axis=1, keepdims=True) + 1e-8
        query_text = " ".join(steady_zone[-96:] if steady_zone else chunks[-1])
        qk, _ = _embed_tokens(query_text.split(), dim, proj, anchors, 0)
        q = _l2norm(qk.mean(axis=0))
        scores: list[tuple[float, int]] = []
        for i, chunk in enumerate(chunks):
            K, _ = _embed_tokens(chunk, dim, proj, anchors, i)
            mu = _l2norm(K.mean(axis=0))
            scores.append((float(mu @ q), i))
        ids = sorted(i for _, i in sorted(scores, reverse=True)[:retrieval_chunks])
        return [chunks[i] for i in ids], ids

    def _select_episodic_chunks(self, chunks: list[list[str]], retrieval_chunks: int,
                                steady_zone: list[str]) -> tuple[list[list[str]], list[int]]:
        if not chunks or retrieval_chunks <= 0:
            return [], []
        import numpy as np
        from .datasets import _embed_tokens, _hash_proj
        from .episode_field import _l2norm, temporal_phase

        dim = int(self.episodic_config.get("dim", 48))
        seed = int(self.episodic_config.get("seed", 0))
        episode_scale = float(self.episodic_config.get("episode_scale", 16.0))
        lam = float(self.episodic_config.get("lam", 0.8))
        recency_bias = float(self.episodic_config.get("recency_bias", 0.15))
        proj = _hash_proj(dim, seed)
        rng = np.random.default_rng(seed)
        anchors = rng.standard_normal((max(len(chunks), 1), dim))
        anchors /= np.linalg.norm(anchors, axis=1, keepdims=True) + 1e-8

        query_text = " ".join(steady_zone[-96:] if steady_zone else chunks[-1])
        qk, _ = _embed_tokens(query_text.split(), dim, proj, anchors, 0)
        q = _l2norm(qk.mean(axis=0))
        active_phase = temporal_phase(np.array([len(chunks)]), r=8, episode_scale=episode_scale)[0]
        scores: list[tuple[float, int]] = []
        for i, chunk in enumerate(chunks):
            K, _ = _embed_tokens(chunk, dim, proj, anchors, i)
            mu = _l2norm(K.mean(axis=0))
            phase = temporal_phase(np.array([i]), r=8, episode_scale=episode_scale)[0]
            semantic = float(mu @ q)
            temporal = 1.0 / (1.0 + float(np.linalg.norm(phase - active_phase)))
            recency = i / max(len(chunks) - 1, 1)
            score = semantic + lam * temporal + recency_bias * recency
            scores.append((score, i))
        ids = sorted(i for _, i in sorted(scores, reverse=True)[:retrieval_chunks])
        return [chunks[i] for i in ids], ids

    def _clean_answer(self, text: str) -> str:
        text = text.strip()
        splitters = ["\nHuman:", "\nUser:", "\nQuestion:", "\nContext:", "You are an AI assistant", "Human:", "User:"]
        for sp in splitters:
            if sp in text:
                text = text.split(sp, 1)[0].strip()
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r"(?:Answer|A)\s*:\s*", "", text, count=1).strip()
        if "." in text:
            first = text.split(".", 1)[0].strip()
            if 1 <= len(first.split()) <= 24:
                return first
        return " ".join(text.split()[:24]).strip()

    def generate(self, session: GenerationSession, *, max_new_tokens: int = 64) -> dict[str, Any]:
        import torch

        zone = self._split_prompt_zones(session.prompt, session.method)
        prompt = zone["prompt"]
        session.steady_words = int(zone["steady_words"])
        session.retrieval_words = int(zone["retrieval_words"])
        session.estimation_words = int(zone["estimation_words"])
        session.metadata["selected_chunk_ids"] = list(zone["selected_chunk_ids"])

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True,
                                max_length=self.max_input_tokens)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        t0 = time.perf_counter()
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=min(max_new_tokens, 24),
                min_new_tokens=1,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                repetition_penalty=1.02,
                no_repeat_ngram_size=4,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
                return_dict_in_generate=False,
            )
        total = time.perf_counter() - t0
        new_tokens = int(out.shape[1] - inputs["input_ids"].shape[1])
        raw_text = self.tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        text = self._clean_answer(raw_text)

        session.generated_text = text
        session.generated_tokens = new_tokens
        gen_pages = self.page_allocator.allocate_tokens(
            session.session_id,
            new_tokens,
            scope="private",
            meta={"kind": "generation", "method": session.method},
            generated=True,
        )
        session.page_ids.extend([p.page_id for p in gen_pages])
        session.metadata["raw_total_latency_s"] = total
        session.metadata["effective_prompt_tokens"] = int(inputs["input_ids"].shape[1])
        session.metadata["raw_generated_text"] = raw_text
        return {
            "text": text,
            "generated_tokens": new_tokens,
            "total_latency_s": total,
        }

    def collect_session_stats(self, session_id: str) -> dict[str, Any]:
        sess = self.sessions[session_id]
        stats = self.page_allocator.session_stats(session_id)
        stats.update({
            "prefix_hit": sess.prefix_hit,
            "prefix_key": sess.prefix_key,
            "method": sess.method,
            "dataset": sess.dataset,
            "generated_tokens": sess.generated_tokens,
            "steady_words": sess.steady_words,
            "retrieval_words": sess.retrieval_words,
            "estimation_words": sess.estimation_words,
        })
        return stats

    def free_session(self, session_id: str) -> None:
        self.page_allocator.free_session(session_id)
        self.sessions.pop(session_id, None)
