"""Adaptive three-tier scheduling for EpisodicKV (VLDB system design Sec. 4).

Short sequences (<500 tokens): disable EpisodicKV, fall back to steady-only.
Medium (500–8k): light mode — fewer k-means iterations, simplified conflict.
Long (>8k): full mode — all three primitives enabled.
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum


class EpisodicMode(str, Enum):
    OFF = "off"
    LIGHT = "light"
    FULL = "full"


@dataclass(frozen=True)
class ModeConfig:
    """Per-mode tuning knobs passed into EpisodicKVCache."""
    enabled: bool
    kmeans_iters: int
    use_attention_jump: bool
    rho: float
    conflict_kappa: float


# VLDB design thresholds (tokens)
SHORT_THRESHOLD = 500
LONG_THRESHOLD = 8000

_MODE_CONFIGS = {
    EpisodicMode.OFF: ModeConfig(
        enabled=False, kmeans_iters=0, use_attention_jump=False, rho=0.0, conflict_kappa=2.0),
    EpisodicMode.LIGHT: ModeConfig(
        enabled=True, kmeans_iters=5, use_attention_jump=False, rho=2.0, conflict_kappa=2.5),
    EpisodicMode.FULL: ModeConfig(
        enabled=True, kmeans_iters=10, use_attention_jump=True, rho=4.0, conflict_kappa=2.0),
}


class AdaptiveScheduler:
    """Maps sequence length to EpisodicMode and returns runtime config."""

    def __init__(self, short: int = SHORT_THRESHOLD, long: int = LONG_THRESHOLD):
        self.short = short
        self.long = long

    def mode_for_length(self, seq_len: int) -> EpisodicMode:
        if seq_len < self.short:
            return EpisodicMode.OFF
        if seq_len <= self.long:
            return EpisodicMode.LIGHT
        return EpisodicMode.FULL

    def config_for_length(self, seq_len: int) -> ModeConfig:
        return _MODE_CONFIGS[self.mode_for_length(seq_len)]

    def config_for_mode(self, mode: EpisodicMode) -> ModeConfig:
        return _MODE_CONFIGS[mode]
