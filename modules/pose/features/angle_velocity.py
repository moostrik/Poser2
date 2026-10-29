from enum import IntEnum
import numpy as np
from .angles import AngleLandmark
from .base.base_scalar_feature import BaseScalarFeature

class AngleVelocity(BaseScalarFeature[AngleLandmark]):
    """Angular velocities for body landmarks (radians/sec, unbounded)."""

    @classmethod
    def enum(cls) -> type[IntEnum]:
        return AngleLandmark

    @classmethod
    def range(cls) -> tuple[float, float]:
        # Unbounded range for velocities
        return (-np.inf, np.inf)

    @classmethod
    def display_range(cls) -> tuple[float, float]:
        """Clamp display to ±π for meaningful visualization."""
        return (-np.pi, np.pi)