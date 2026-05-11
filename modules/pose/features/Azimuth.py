from __future__ import annotations

from .base import NormalizedSingleValue


class Azimuth(NormalizedSingleValue):
    """Horizontal world-space position normalized to [0, 1] over 360°.

    Populated by the panoramic tracker. Absent (NaN, score 0.0) for other tracker types.
    """

    @classmethod
    def range(cls) -> tuple[float, float]:
        return (0.0, 1.0)
