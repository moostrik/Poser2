from dataclasses import replace
import numpy as np

from modules.pose.Pose import Pose, PoseDict
from modules.pose.features.PosePoints import PosePointData
from modules.pose.features.PoseAngles import PoseAngleData
from modules.pose.filters.PoseFilterBase import PoseFilterBase

from modules.Settings import Settings


class PoseDeltaFilter(PoseFilterBase):
    """Computes frame-to-frame changes (deltas) for pose data.

    Calculates:
    - Point displacement: Image space movement (pixels) since last frame
    - Angle displacement: Angular change (with proper wrapping) since last frame

    Handles occlusion: Sets deltas to NaN when joints reappear after being invalid.
    """

    def __init__(self) -> None:
        super().__init__()
        self._prev_poses: dict[int, Pose] = {}  # tracklet_id -> previous Pose

    def add_poses(self, poses: PoseDict) -> None:
        """Compute deltas for all poses and emit enriched results."""
        enriched_poses: PoseDict = {}

        for pose_id, pose in poses.items():
            tracklet_id: int = pose.tracklet.id

            # Get previous pose for this tracklet
            prev_pose: Pose | None = self._prev_poses.get(tracklet_id)

            # Compute deltas (or empty if no previous pose)
            if prev_pose is None:
                point_delta_data: PosePointData = PosePointData.create_empty()
                angle_delta_data: PoseAngleData = PoseAngleData.create_empty()
            else:
                point_delta_data = pose.camera_points.subtract(prev_pose.camera_points)
                angle_delta_data = pose.angle_data.subtract(prev_pose.angle_data)

            # Update state for next frame
            self._prev_poses[tracklet_id] = pose

            # Create enriched pose
            enriched_pose: Pose = replace(
                pose,
                point_delta_data=point_delta_data,
                angle_delta_data=angle_delta_data
            )

            enriched_poses[pose_id] = enriched_pose

            # Cleanup lost tracklets
            if pose.lost:
                del self._prev_poses[tracklet_id]

        self._notify_callbacks(enriched_poses)