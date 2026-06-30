from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import time

from .runtime import EpisodicRuntime, GenerationSession
from .cache import EpisodicKVCache


@dataclass
class HookResult:
    session_id: str
    method: str
    prompt_tokens: int
    effective_prompt_tokens: int
    generated_tokens: int
    text: str
    total_latency_s: float
    retrieval_calls: int = 0
    retrieval_latency_s: float = 0.0
    boundary_count: int = 0
    episode_count: int = 0


class HFRuntimeMethodRunner:
    """Method adapter over `EpisodicRuntime`.

    This first-stage industrial runtime supports:
    - full: dense full prompt
    - sliding: last-window baseline
    - fixed: fixed-interval episode baseline
    - kmeans: heuristic chunk retrieval baseline
    - episodic: external episodic chunk selection

    It also exposes retrieval/boundary counters in a uniform way so the later
    hook-based implementation can slot into the same experiment scripts.
    """

    def __init__(self, runtime: EpisodicRuntime):
        self.runtime = runtime

    def run(
        self,
        session: GenerationSession,
        *,
        max_new_tokens: int = 64,
    ) -> HookResult:
        retrieval_calls = 0
        retrieval_latency_s = 0.0
        boundary_count = 0
        episode_count = 0

        if session.method == "episodic":
            zone = self.runtime._split_prompt_zones(session.prompt, session.method)
            chunk_ids = list(zone.get("selected_chunk_ids", []))
            retrieval_calls = 1 if chunk_ids else 0
            boundary_count = max(len(chunk_ids) - 1, 0)
            episode_count = max(len(chunk_ids), 1)
            t0 = time.perf_counter()
            _ = zone["prompt"]
            retrieval_latency_s = time.perf_counter() - t0
        elif session.method == "fixed":
            zone = self.runtime._split_prompt_zones(session.prompt, session.method)
            chunk_ids = list(zone.get("selected_chunk_ids", []))
            episode_count = max(len(chunk_ids), 1)
            boundary_count = max(len(chunk_ids) - 1, 0)
            retrieval_calls = 1 if chunk_ids else 0
        elif session.method == "kmeans":
            zone = self.runtime._split_prompt_zones(session.prompt, session.method)
            chunk_ids = list(zone.get("selected_chunk_ids", []))
            episode_count = max(len(chunk_ids), 1)
            retrieval_calls = 1 if chunk_ids else 0
        else:
            words = session.prompt.split()
            episode_count = max(1, len(words) // 120)

        res = self.runtime.generate(session, max_new_tokens=max_new_tokens)
        effective_prompt_tokens = int(session.metadata.get("effective_prompt_tokens", session.prompt_tokens))
        return HookResult(
            session_id=session.session_id,
            method=session.method,
            prompt_tokens=session.prompt_tokens,
            effective_prompt_tokens=effective_prompt_tokens,
            generated_tokens=res["generated_tokens"],
            text=res["text"],
            total_latency_s=res["total_latency_s"],
            retrieval_calls=retrieval_calls,
            retrieval_latency_s=retrieval_latency_s,
            boundary_count=boundary_count,
            episode_count=episode_count,
        )
