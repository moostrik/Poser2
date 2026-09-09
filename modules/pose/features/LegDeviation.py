from __future__ import annotations

from .base import NormalizedSingleValue


class LegDeviation(NormalizedSingleValue):
    """How far the legs are from standing straight, in [0, 1].

    Joint-weighted deviation of the hip and knee angles from neutral (the knee flexes
    further for the same effort, so it is normalised by a larger angle), aggregated over
    the most-bent joints so one bent leg registers fully. Populated by
    LegDeviationExtractor. Absent (NaN, score 0.0) when no leg angle is valid.
    """

    @classmethod
    def range(cls) -> tuple[float, float]:
        return (0.0, 1.0)
