import numpy as np
from modules.pose.features.Angles import AngleLandmark
from modules.pose.features.base.BaseScalarFeature import BaseScalarFeature

class AngleVelocity(BaseScalarFeature[AngleLandmark]):
    """Angular velocities for body landmarks (radians/sec, unbounded)."""

    @classmethod
    def feature_enum(cls) -> type[AngleLandmark]:
        return AngleLandmark

    @classmethod
    def default_range(cls) -> tuple[float, float]:
        # Unbounded range for velocities
        return (-np.inf, np.inf)