"""NeutralGate — a pair's similarity is 0 while either person stands neutral.

Sits between ``WindowSimilarity`` and the ``SimilarityApplicator``, so the gate acts before the frames are
stamped and before the ``Similarity`` smoothers (Euro at SMOOTH, sticky at PREDICT, chase at LERP) and the
rate limiter: the gate is hard, the pipeline makes it soft in time, and a pose crossing neutral never jumps
the sync. Neutral is arms hanging: ``ArmDeviation`` at or below ``neutral`` (``docs/STATES.md``,
*Vocabulary*). Every consumer of ``Similarity`` — the state machine's sync, the pose instrument's window
opening, the sound — sees the gated value.

Threads: ``set_frames`` runs on the pose input thread (the SMOOTH stage broadcast), ``process`` on the
analytics thread; the arm deviations are handed over under a lock.
"""

from __future__ import annotations

import math
from threading import Lock
from typing import Callable

import numpy as np

from modules.pose import FrameDict, features
from modules.pose.analytics import SimilarityResult
from modules.settings import BaseSettings, Field


class NeutralGateSettings(BaseSettings):
    """Where neutral ends, and whether the gate is on."""
    enabled: Field[bool]  = Field(True, description="Zero a pair's similarity while either person stands neutral")
    neutral: Field[float] = Field(0.1, min=0.0, max=1.0, step=0.01, description="Arm deviation up to which a person stands neutral")


class NeutralGate:
    """Gates the pairwise similarity rows by the participants' arm deviation; see the module docstring."""

    def __init__(self, config: NeutralGateSettings) -> None:
        self._config = config
        self._lock = Lock()
        self._deviation: dict[int, float] = {}
        self._callbacks: list[Callable[[SimilarityResult], None]] = []

    def add_similarity_callback(self, callback: Callable[[SimilarityResult], None]) -> None:
        self._callbacks.append(callback)

    def set_frames(self, frames: FrameDict) -> None:
        """Remember each present person's arm deviation (pose input thread)."""
        deviation = {id: frame[features.ArmDeviation].value for id, frame in frames.items()}
        with self._lock:
            self._deviation = deviation

    def process(self, result: SimilarityResult) -> None:
        """Gate the pairwise rows and emit them (analytics thread)."""
        if self._config.enabled:
            result = self._gate(result)
        for callback in self._callbacks:
            callback(result)

    def _gate(self, result: SimilarityResult) -> SimilarityResult:
        with self._lock:
            deviation = self._deviation
        neutral = self._config.neutral
        length = features.Similarity.length()
        # 1.0 at the ids of the people posing (arms seen and past neutral), 0.0 elsewhere.
        posing = np.zeros(length, dtype=np.float32)
        for id, value in deviation.items():
            if id < length and not math.isnan(value) and value > neutral:
                posing[id] = 1.0

        rows: dict[int, features.Similarity] = {}
        for id, row in result.similarity.items():
            gate = posing if id < length and posing[id] > 0.0 else np.zeros(length, dtype=np.float32)
            rows[id] = features.Similarity(row.values * gate, row.scores)   # a gated pair is a real 0; NaN stays NaN
        return SimilarityResult(rows, result.leader_score)
