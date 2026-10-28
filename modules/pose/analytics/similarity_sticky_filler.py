"""SimilarityStickyFiller — holds a present pair's last similarity across a frame without one.

Runs on the analytics result, before the rows are stamped on the poses, because that is where presence is
known: the result's tracks are the poses the analytics saw. A slot is held only while its player is in the
result; a departed player's slot is NaN (no data) and its memory is dropped, so nothing downstream carries a
similarity toward someone who is gone, and a track that leaves and returns starts clean. The self slot stays
NaN. Runs on the analytics thread only.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from ..features import Similarity
from .window_similarity import SimilarityResult
from modules.settings import BaseSettings, Field


class SimilarityStickyFillerSettings(BaseSettings):
    """Whether a present pair's gap is bridged, and with which score."""
    enabled:     Field[bool] = Field(True,  description="Hold a present pair's last similarity across a frame without one")
    hold_scores: Field[bool] = Field(False, description="A held value keeps its last score (else 0)")


class SimilarityStickyFiller:
    """Fills a present pair's NaN slot from its last valid value; see the module docstring."""

    def __init__(self, config: SimilarityStickyFillerSettings | None = None) -> None:
        self._config = config if config is not None else SimilarityStickyFillerSettings()
        self._values: dict[int, np.ndarray] = {}      # per track: the last valid value per slot (NaN: none)
        self._scores: dict[int, np.ndarray] = {}
        self._callbacks: list[Callable[[SimilarityResult], None]] = []

    def add_similarity_callback(self, callback: Callable[[SimilarityResult], None]) -> None:
        self._callbacks.append(callback)

    def process(self, result: SimilarityResult) -> None:
        """Fill the rows' gaps and emit the result."""
        if self._config.enabled:
            result = self._fill(result)
        for callback in self._callbacks:
            callback(result)

    def reset(self) -> None:
        """Forget every held value."""
        self._values.clear()
        self._scores.clear()

    def _fill(self, result: SimilarityResult) -> SimilarityResult:
        length = Similarity.length()
        present = np.zeros(length, dtype=bool)
        for tid in result.similarity:
            if tid < length:
                present[tid] = True
        for tid in list(self._values):                 # a track that left starts clean when it returns
            if tid not in result.similarity:
                del self._values[tid], self._scores[tid]

        hold_scores = self._config.hold_scores
        rows: dict[int, Similarity] = {}
        for tid, row in result.similarity.items():
            last_values = self._values.get(tid)
            last_scores = self._scores.get(tid)
            if last_values is None or last_scores is None:
                last_values = np.full(length, np.nan, dtype=np.float32)
                last_scores = np.zeros(length, dtype=np.float32)

            values = row.values.astype(np.float32, copy=True)
            scores = row.scores.astype(np.float32, copy=True)
            gap = np.isnan(values) & present & ~np.isnan(last_values)   # a present partner without a value this frame
            if tid < length:
                gap[tid] = False                                        # the self slot is never a pair
            values[gap] = last_values[gap]
            scores[gap] = last_scores[gap] if hold_scores else 0.0

            valid = ~np.isnan(row.values)
            new_values = np.where(valid, row.values, last_values).astype(np.float32)
            new_scores = np.where(valid, row.scores, last_scores).astype(np.float32)
            new_values[~present] = np.nan                               # a departed partner is forgotten
            new_scores[~present] = 0.0
            self._values[tid] = new_values
            self._scores[tid] = new_scores
            rows[tid] = Similarity(values, scores)
        return SimilarityResult(rows, result.leader_score)
