# Standard library imports
from dataclasses import replace

# Pose imports
from modules.pose.filters.PoseFilterBase import PoseFilterBase
from modules.pose.features.PoseAngles import PoseAngleData, PoseAngleFactory
from modules.pose.Pose import Pose, PoseDict


class PoseAngleExtractor(PoseFilterBase):
    """Computes joint angles from pose keypoint data.

    Calculates:
    - Joint angles: Angular measurements at body joints (shoulders, elbows, hips, knees)
    - Head yaw: Head rotation relative to torso

    Uses PoseAngleFactory to compute angles from 2D keypoint positions with proper
    rotation offsets and symmetric mirroring for right-side joints.
    """

    def add_poses(self, poses: PoseDict) -> None:
        """Compute angles for all poses and emit enriched results."""
        enriched_poses: PoseDict = {}

        for pose_id, pose in poses.items():
            # Compute angles from camera points
            angle_data: PoseAngleData = PoseAngleFactory.from_points(pose.camera_points)

            # Create enriched pose
            enriched_pose: Pose = replace(
                pose,
                angle_data=angle_data
            )

            enriched_poses[pose_id] = enriched_pose

        self._notify_callbacks(enriched_poses)