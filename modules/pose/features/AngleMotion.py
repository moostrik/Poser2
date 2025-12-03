from enum import IntEnum
import numpy as np
from modules.pose.features.Angles import AngleLandmark
from modules.pose.features import NormalizedScalarFeature


ANGLE_MOTION_NORMALISATION: dict[AngleLandmark, float] = {
    AngleLandmark.left_shoulder:   1.0,
    AngleLandmark.right_shoulder:  1.0,
    AngleLandmark.left_elbow:      1.0,
    AngleLandmark.right_elbow:     1.0,
    AngleLandmark.left_hip:        2.0,
    AngleLandmark.right_hip:       2.0,
    AngleLandmark.left_knee:       2.0,
    AngleLandmark.right_knee:      2.0,
    AngleLandmark.head:            3.0,
}



class AngleMotion(NormalizedScalarFeature[AngleLandmark]):
    """Angular velocities for body landmarks (radians/sec, unbounded)."""

    @classmethod
    def enum(cls) -> type[IntEnum]:
        return AngleLandmark