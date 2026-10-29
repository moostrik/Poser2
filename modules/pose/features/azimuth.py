from __future__ import annotations

from .base import SingleAngle


class Azimuth(SingleAngle):
    """A person's horizontal world-space angle at their eyes, in radians [-π, π).

    Derived from ``BBoxAzimuth`` by ``AzimuthExtractor``. Absent (NaN, score 0.0) without one.
    """
