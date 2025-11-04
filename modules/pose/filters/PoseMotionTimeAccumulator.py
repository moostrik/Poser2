from dataclasses import replace
import numpy as np

from modules.pose.Pose import Pose, PoseDict
from modules.pose.filters.PoseFilterBase import PoseFilterBase


class PoseMotionTimeAccumulator(PoseFilterBase):
    """Takes the absolute value of all deltas and adds them to the movement_time field."""


    def add_poses(self, poses: PoseDict) -> None:
        """Compute deltas for all poses and emit enriched results."""
        enriched_poses: PoseDict = {}
        for pose_id, pose in poses.items():
            # Compute movement time as sum of absolute deltas
            total_delta: float = np.nansum(np.abs(pose.delta_data.values))
            motion_time: float = pose.motion_time + total_delta

            # Create enriched pose
            enriched_pose: Pose = replace(
                pose,
                motion_time=motion_time
            )

            enriched_poses[pose_id] = enriched_pose

        self._notify_callbacks(enriched_poses)