from __future__ import annotations

from .base import NormalizedSingleValue


class ArmDeviation(NormalizedSingleValue):
    """How far the arms are out of neutral, in [0, 1].

    The position of the arms' movement within the extractor's degree range: 0 while the
    most-moved arm joint (shoulder or elbow, calibrated so hanging / straight is 0) is within
    ``min_degrees`` of neutral, 1 from ``max_degrees`` on, linear between. Populated by
    ArmDeviationExtractor. Absent (NaN, score 0.0) when no arm angle is valid.
    """

    @classmethod
    def range(cls) -> tuple[float, float]:
        return (0.0, 1.0)
