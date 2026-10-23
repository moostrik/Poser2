from __future__ import annotations

from .base import NormalizedSingleValue


class LegDeviation(NormalizedSingleValue):
    """How far the legs are from standing straight, in [0, 1].

    Each hip and knee angle's position in the extractor's degree range (0 up to a shared
    minimum bend, 1 from the joint's own top — the knee flexes further for the same effort,
    so its top is larger), aggregated over the most-bent joints so one bent leg registers
    fully. Populated by LegDeviationExtractor. Absent (NaN, score 0.0) when no leg angle is valid.
    """

    @classmethod
    def range(cls) -> tuple[float, float]:
        return (0.0, 1.0)
