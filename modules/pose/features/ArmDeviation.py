from __future__ import annotations

from .base import NormalizedSingleValue


class ArmDeviation(NormalizedSingleValue):
    """How far the arms are from hanging, in [0, 1].

    Joint-weighted deviation of the shoulder and elbow angles from the calibrated neutral
    (0 hanging, 0 straight), each normalised by its own full-deviation angle and aggregated
    over the most-deviated joints so one raised arm registers fully. Populated by
    ArmDeviationExtractor. Absent (NaN, score 0.0) when no arm angle is valid.
    """

    @classmethod
    def range(cls) -> tuple[float, float]:
        return (0.0, 1.0)
