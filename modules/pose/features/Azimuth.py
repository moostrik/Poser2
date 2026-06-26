from __future__ import annotations

from .base import SingleAngle


class Azimuth(SingleAngle):
    """Horizontal world-space angular position in radians [-π, π).

    Populated by the panoramic tracker. Absent (NaN, score 0.0) for other tracker types.
    """
