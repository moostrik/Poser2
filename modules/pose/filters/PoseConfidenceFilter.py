from dataclasses import replace

import numpy as np

from modules.pose.Pose import Pose, PoseDict
from modules.pose.features.PosePoints import PosePointData
from modules.pose.filters.PoseFilterBase import PoseFilterBase
from modules.Settings import Settings


class PoseConfidenceFilter(PoseFilterBase):
    """Filters pose keypoints based on confidence thresholds.

    Removes low-confidence keypoints by:
    1. Setting values to NaN when score < threshold
    2. Rescaling remaining scores to [0, 1] range

    Note: Does NOT recompute angles - that should be done by PoseAngles filter downstream.
    """

    def __init__(self, settings: Settings) -> None:
        super().__init__()
        self.confidence_threshold: float = max(0.0, min(0.99, settings.pose_conf_threshold))

    def add_poses(self, poses: PoseDict) -> None:
        """Filter low-confidence keypoints from all poses."""
        filtered_poses: PoseDict = {}
        for pose_id, pose in poses.items():
            if pose.point_data.valid_count == 0:
                filtered_poses[pose_id] = pose
                continue

            values: np.ndarray = pose.point_data.values
            scores: np.ndarray = pose.point_data.scores

            # Mask for keypoints meeting confidence threshold
            filtered = scores >= self.confidence_threshold

            # Replace low-confidence keypoints with NaN
            filtered_values: np.ndarray = np.where(filtered[:, np.newaxis], values, np.nan)

            # Rescale remaining scores to [0, 1] range
            rescaled_scores: np.ndarray = np.where(filtered, (scores - self.confidence_threshold) / (1 - self.confidence_threshold), 0.0)

            filtered_points = PosePointData(values=filtered_values, scores=rescaled_scores)
            filtered_pose: Pose = replace(pose, point_data=filtered_points)
            filtered_poses[pose_id] = filtered_pose

        # Notify callbacks with filtered poses
        self._notify_callbacks(filtered_poses)
