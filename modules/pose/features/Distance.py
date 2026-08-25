from __future__ import annotations

from .base import NormalizedSingleValue


class Distance(NormalizedSingleValue):
    """Normalized distance estimate in [0, 1] derived from a person's vertical position.

    Approximates how far a person is from the camera by mapping the bbox bottom
    edge (feet) between two y thresholds: closest → 0.0, farthest → 1.0.
    Populated by DistanceExtractor. Absent (NaN, score 0.0) when not extracted.
    """

    @classmethod
    def range(cls) -> tuple[float, float]:
        return (0.0, 1.0)
