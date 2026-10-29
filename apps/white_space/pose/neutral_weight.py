"""NeutralWeight — a pair's similarity weighted by how far both people are out of neutral.

A group step between the ``PostureSimilarityExtractor`` and the ``Similarity`` smoothers (Euro at SMOOTH, chase
at LERP), so the weight acts before the rows are smoothed. The weight is the ``ArmDeviation`` the arm extractor
measures on the same frame — 0 with the arms within its ``min_degrees`` of hanging (neutral, ``docs/STATES.md``
*Vocabulary*), 1 from its ``max_degrees`` — and a pair reads as its member closer to neutral: the minimum of
the two. This node has no thresholds of its own; it only multiplies. The deviation's range handles a pose
hovering near neutral, the smoothers handle speed, so a pose leaving neutral never jumps the level. Every
consumer of the ``Similarity`` feature — the pose instrument's window opening, the sound, the data view — sees
the weighted value: a ramp. The hit sync does not read the feature; it gates on the same ``ArmDeviation``
being 1 (``hit_sync.py``), under the same ``enabled`` switch.
"""

from __future__ import annotations

import numpy as np

from modules.pose import FrameDict, features, nodes, replace
from modules.settings import BaseSettings, Field


class NeutralWeightSettings(BaseSettings):
    """Whether the weight is applied."""
    enabled: Field[bool] = Field(True, description="Weight a pair's similarity by how far both are out of neutral")


class NeutralWeight(nodes.GroupNode):
    """Weights the pairwise similarity rows by the players' arm deviation; see the module docstring."""

    def __init__(self, settings: NeutralWeightSettings) -> None:
        self._settings = settings

    def process(self, frames: FrameDict) -> FrameDict:
        """Weight every pose's row by its and its partners' arm deviation, from this frame."""
        if not self._settings.enabled:
            return frames
        length = features.Similarity.length()
        # Each present person's deviation at their id; 0 elsewhere and for arms not seen (NaN): neutral.
        weights = np.zeros(length, dtype=np.float32)
        for id, frame in frames.items():
            value = frame[features.ArmDeviation].value
            if id < length and not np.isnan(value):
                weights[id] = value

        out: FrameDict = {}
        for id, frame in frames.items():
            row = frame[features.Similarity]
            own = weights[id] if id < length else 0.0
            pair = np.minimum(own, weights)                       # a pair reads as its less-moved member
            out[id] = replace(frame, {features.Similarity: features.Similarity(row.values * pair, row.scores)})   # a weighted pair is a real value; NaN stays NaN
        return out
