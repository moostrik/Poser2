from __future__ import annotations

from .base import NormalizedSingleValue


class GroupSimilarity(NormalizedSingleValue):
    """How alike the group is, in [0, 1]: the level ``analytics.group_similarity_level`` gives the frame's poses
    by the gate's mode (the least alike pair, or the best level with one left out). A group record feature,
    written by ``GroupSyncGate``. Absent (NaN, score 0.0) with fewer than two poses or a pair that cannot be
    read."""

    @classmethod
    def range(cls) -> tuple[float, float]:
        return (0.0, 1.0)
