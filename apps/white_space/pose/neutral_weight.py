"""NeutralWeight — a pair's similarity weighted by how far both people are out of neutral.

Sits between ``WindowSimilarity`` and the ``SimilarityApplicator``, so the weight acts before the frames are
stamped and before the ``Similarity`` smoothers (Euro at SMOOTH, sticky at PREDICT, chase at LERP). The
weight is the ``ArmDeviation`` the arm extractor measures — 0 with the arms within its ``min_degrees`` of
hanging (neutral, ``docs/STATES.md`` *Vocabulary*), 1 from its ``max_degrees`` — and a pair reads as its
less-moved member: the minimum of the two. This node has no thresholds of its own; it only multiplies.
The deviation's range handles a pose hovering near neutral, the smoothers handle speed, so a pose leaving
neutral never jumps the sync. Every consumer of ``Similarity`` — the state machine's sync, the pose
instrument's window opening, the sound — sees the weighted value.

Threads: ``set_frames`` runs on the pose input thread (the SMOOTH stage broadcast), ``process`` on the
analytics thread; the deviations are handed over under a lock.
"""

from __future__ import annotations

from threading import Lock
from typing import Callable

import numpy as np

from modules.pose import FrameDict, features
from modules.pose.analytics import SimilarityResult
from modules.settings import BaseSettings, Field


class NeutralWeightSettings(BaseSettings):
    """Whether the weight is applied."""
    enabled: Field[bool] = Field(True, description="Weight a pair's similarity by how far both are out of neutral")


class NeutralWeight:
    """Weights the pairwise similarity rows by the players' arm deviation; see the module docstring."""

    def __init__(self, config: NeutralWeightSettings) -> None:
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
        """Weight the pairwise rows and emit them (analytics thread)."""
        if self._config.enabled:
            result = self._weight(result)
        for callback in self._callbacks:
            callback(result)

    def _weight(self, result: SimilarityResult) -> SimilarityResult:
        with self._lock:
            deviation = self._deviation
        length = features.Similarity.length()
        # Each present person's deviation at their id; 0 elsewhere and for arms not seen (NaN): neutral.
        weights = np.zeros(length, dtype=np.float32)
        for id, value in deviation.items():
            if id < length and not np.isnan(value):
                weights[id] = value

        rows: dict[int, features.Similarity] = {}
        for id, row in result.similarity.items():
            own = weights[id] if id < length else 0.0
            pair = np.minimum(own, weights)                       # a pair reads as its less-moved member
            rows[id] = features.Similarity(row.values * pair, row.scores)   # a weighted pair is a real value; NaN stays NaN
        return SimilarityResult(rows, result.leader_score)
