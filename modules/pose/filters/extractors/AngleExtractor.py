# Standard library imports
from dataclasses import replace

# Pose imports
from modules.pose.filters.FilterBase import FilterBase
from modules.pose.features.PoseAngles import PoseAngleData, PoseAngleFactory
from modules.pose.Pose import Pose


class AngleExtractor(FilterBase):
    """Computes joint angles from pose keypoint data.

    Calculates:
    - Joint angles: Angular measurements at body joints (shoulders, elbows, hips, knees)
    - Head yaw: Head rotation relative to torso

    Uses PoseAngleFactory to compute angles from 2D keypoint positions with proper
    rotation offsets and symmetric mirroring for right-side joints.
    """

    def process(self, pose: Pose) -> Pose:
        """Compute angles for all poses and emit enriched results."""
        angle_data: PoseAngleData = PoseAngleFactory.from_points(pose.camera_points)
        return replace(pose, angle_data=angle_data)