from __future__ import annotations

from .base import SingleValue

TORSO_TILT_RANGE: tuple[float, float] = (-1.0, 1.0)


class TorsoTilt(SingleValue):
    """Sideways lean of the torso against the image vertical, signed, in [-1, 1].

    The one pose measure taken against an absolute axis (every joint angle is measured
    between body segments): the angle of the spine — hip midpoint to shoulder midpoint —
    from the image's vertical, normalised by the extractor's ``tilt_degrees``. 0 = upright;
    positive = shoulders displaced toward image right of the hips. Populated by
    TorsoTiltExtractor. Absent (NaN, score 0.0) when the spine keypoints are missing.
    """

    @classmethod
    def range(cls) -> tuple[float, float]:
        return TORSO_TILT_RANGE
