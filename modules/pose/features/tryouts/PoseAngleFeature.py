from abc import ABC
import numpy as np
from typing import Optional


from typing_extensions import Self

from modules.pose.features.tryouts.PoseVectorFeatureBase import PoseVectorFeatureBase, PoseEnum


class PoseAngleFeature(PoseVectorFeatureBase[PoseEnum], ABC):
    """Base class for angle-based pose features.

    Extends PoseFeatureBase with angle-specific functionality:
    - Enforces angle range [-π, π]
    - Provides angular distance calculations

    Used by:
    - PoseAngles: Joint angles
    - PoseAngleSymmetry: Bilateral symmetry measurements
    """

    @classmethod
    def default_range(cls) -> tuple[float, float]:
        """Angles must be in range [-π, π]."""
        return (-np.pi, np.pi)

    def angular_distance(self, other: 'PoseAngleFeature', joint: PoseEnum | int) -> float:
        """Calculate angular distance between two angles at a joint. """
        angle1 = self._values[joint]
        angle2 = other._values[joint]

        # Return NaN if either angle is invalid
        if np.isnan(angle1) or np.isnan(angle2):
            return np.nan

        # Calculate shortest angular distance
        diff = angle1 - angle2
        # Normalize difference to [-π, π]
        normalized_diff = np.arctan2(np.sin(diff), np.cos(diff))
        # Return absolute value (distance is always positive)
        return float(np.abs(normalized_diff))

    def angular_distances(self, other: 'PoseAngleFeature') -> np.ndarray:
        """Calculate angular distances for all joints."""
        # Calculate differences
        diff = self._values - other._values

        # Normalize to [-π, π]
        normalized_diff = np.arctan2(np.sin(diff), np.cos(diff))

        # Return absolute values
        return np.abs(normalized_diff)

    def mean_angular_distance(self, other: 'PoseAngleFeature') -> float:
        """Calculate mean angular distance across all valid joints."""
        distances = self.angular_distances(other)
        return float(np.nanmean(distances))

    def subtract(self, other: Self) -> Self:
        """Compute angular differences with proper wrapping to [-π, π] range."""
        diff = self._values - other._values
        # Wrap angles to [-π, π] range (shortest angular distance)
        wrapped_diff = np.arctan2(np.sin(diff), np.cos(diff))
        min_scores = np.minimum(self._scores, other._scores)
        return type(self)(values=wrapped_diff, scores=min_scores)

    def similarity(self, other: Self, exponent: float = 1.0) -> Self:
        """Compute similarity scores between angle features."""
        diff_data = self.subtract(other)
        similarity_values = np.power(1.0 - np.abs(diff_data._values) / np.pi, exponent)
        return type(self)(values=similarity_values, scores=diff_data._scores)