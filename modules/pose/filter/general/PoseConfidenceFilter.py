# Standard library imports
from dataclasses import replace

# Third-party imports
import numpy as np

# Pose imports
from modules.pose.filter.PoseFilterBase import PoseFilterBase, PoseFilterConfigBase
from modules.pose.features.PosePoints import PosePointData
from modules.pose.Pose import Pose

class PoseConfidenceFilterConfig(PoseFilterConfigBase):
    """Configuration for confidence filtering with automatic change notification."""

    def __init__(self, confidence_threshold: float = 0.5) -> None:
        super().__init__()
        self.confidence_threshold: float = max(0.0, min(0.99, confidence_threshold))


class PoseConfidenceFilter(PoseFilterBase):
    """Filters pose keypoints based on confidence thresholds.

    Removes low-confidence keypoints by:
    1. Setting values to NaN when score < threshold
    2. Rescaling remaining scores to [0, 1] range

    Note: Does NOT recompute angles - that should be done by PoseAngles filter downstream.
    """

    def __init__(self, config: PoseConfidenceFilterConfig) -> None:
        super().__init__()
        self._config: PoseConfidenceFilterConfig = config

    def process(self, pose: Pose) -> Pose:
        """Process a single pose."""
        if pose.point_data.valid_count == 0:
            return pose

        values: np.ndarray = pose.point_data.values
        scores: np.ndarray = pose.point_data.scores

        # Mask for keypoints meeting confidence threshold
        filtered = scores >= self._config.confidence_threshold

        # Replace low-confidence keypoints with NaN
        filtered_values: np.ndarray = np.where(filtered[:, np.newaxis], values, np.nan)

        # Rescale remaining scores to [0, 1] range
        rescaled_scores: np.ndarray = np.where(filtered, (scores - self._config.confidence_threshold) / (1 - self._config.confidence_threshold), 0.0)

        filtered_points = PosePointData(values=filtered_values, scores=rescaled_scores)
        filtered_pose: Pose = replace(pose, point_data=filtered_points)
        return filtered_pose
