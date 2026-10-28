from __future__ import annotations

from .base import NormalizedSingleValue


class Distance(NormalizedSingleValue):
    """How far a person stands from the fixture within the tracked zone, in [0, 1].

    0 at the zone's near edge, 1 at its far edge, clamped. Populated by the panoramic tracker from
    where the feet meet the floor. Absent (NaN, score 0.0) for other tracker types and without a
    reading.
    """

    @classmethod
    def range(cls) -> tuple[float, float]:
        return (0.0, 1.0)
