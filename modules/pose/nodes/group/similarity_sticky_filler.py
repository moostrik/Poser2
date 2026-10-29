"""SimilarityStickyFiller — holds a present pair's last similarity across a frame without one.

A group step on the frames' ``Similarity`` rows, placed before they are smoothed, because that is where
presence is known: the frame's poses are the people present. A slot is held only while its player is in the
frame; a departed player's slot is NaN (no data) and its memory is dropped, so nothing downstream carries a
similarity toward someone who is gone, and a track that leaves and returns starts clean. The self slot stays
NaN. Runs on the frame's thread only.
"""

from __future__ import annotations

import numpy as np

from ...features import Similarity
from ...frame import FrameDict, replace
from ..nodes import GroupNode
from modules.settings import BaseSettings, Field


class SimilarityStickyFillerSettings(BaseSettings):
    """Whether a present pair's gap is bridged, and with which score."""
    enabled:     Field[bool] = Field(True,  description="Hold a present pair's last similarity across a frame without one")
    hold_scores: Field[bool] = Field(False, description="A held value keeps its last score (else 0)")


class SimilarityStickyFiller(GroupNode):
    """Fills a present pair's NaN slot from its last valid value; see the module docstring."""

    def __init__(self, settings: SimilarityStickyFillerSettings | None = None) -> None:
        self.settings = settings if settings is not None else SimilarityStickyFillerSettings()
        self._values: dict[int, np.ndarray] = {}      # per track: the last valid value per slot (NaN: none)
        self._scores: dict[int, np.ndarray] = {}

    def process(self, frames: FrameDict) -> FrameDict:
        """Fill the rows' gaps and return the frames, stamped."""
        if not self.settings.enabled:
            return frames
        return self._fill(frames)

    def reset(self) -> None:
        """Forget every held value."""
        self._values.clear()
        self._scores.clear()

    def _fill(self, frames: FrameDict) -> FrameDict:
        length = Similarity.length()
        present = np.zeros(length, dtype=bool)
        for tid in frames:
            if tid < length:
                present[tid] = True
        for tid in list(self._values):                 # a track that left starts clean when it returns
            if tid not in frames:
                del self._values[tid], self._scores[tid]

        hold_scores = self.settings.hold_scores
        out: FrameDict = {}
        for tid, frame in frames.items():
            row = frame[Similarity]
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
            out[tid] = replace(frame, {Similarity: Similarity(values, scores)})
        return out
