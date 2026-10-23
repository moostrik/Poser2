from __future__ import annotations

from .base import SingleAngle


class BBoxAzimuth(SingleAngle):
    """The tracker's horizontal world-space angle of the detection box, in radians [-π, π).

    Populated by the panoramic tracker; at a seam, its blend of both cameras' views. The input
    ``AzimuthExtractor`` derives ``Azimuth`` from. Absent (NaN, score 0.0) for other tracker types.
    """
