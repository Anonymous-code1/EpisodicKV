"""EpisodicKV: episode-aware dynamic KV cache index for long-context LLMs.

CPU mechanism-validation prototype (numpy only) + real-model hook skeleton.
See PLAN.md for the math.
"""
from .episode_field import EpisodeFieldIndex, temporal_phase
from .conflict import ConflictPotential
from .retrieval import episode_decayed_attention
from .cache import EpisodicKVCache
from .baselines import (
    full_attention, KMeansKVCache, SlidingWindowKVCache,
    FixedIntervalEpisodeCache, RoPETemporalKMeansCache,
)
from .scheduler import AdaptiveScheduler, EpisodicMode
from .storage import DualLayerStorage, GlobalPublicEpisodePool, SessionPrivateEpisodePool
from .paged_pool import KVPage, SessionPageTable, PageAllocator, GlobalPrefixPool
from .runtime import EpisodicRuntime, GenerationSession
from .hf_runtime_hook import HFRuntimeMethodRunner, HookResult

__all__ = [
    "EpisodeFieldIndex",
    "temporal_phase",
    "ConflictPotential",
    "episode_decayed_attention",
    "EpisodicKVCache",
    "full_attention",
    "KMeansKVCache",
    "SlidingWindowKVCache",
    "FixedIntervalEpisodeCache",
    "RoPETemporalKMeansCache",
    "AdaptiveScheduler",
    "EpisodicMode",
    "DualLayerStorage",
    "GlobalPublicEpisodePool",
    "SessionPrivateEpisodePool",
    "KVPage",
    "SessionPageTable",
    "PageAllocator",
    "GlobalPrefixPool",
    "EpisodicRuntime",
    "GenerationSession",
    "HFRuntimeMethodRunner",
    "HookResult",
]