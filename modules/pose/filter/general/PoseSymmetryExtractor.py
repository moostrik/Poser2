# Standard library imports
from dataclasses import replace

# Pose imports
from modules.pose.filter.PoseFilterBase import PoseFilterBase
from modules.pose.features.PoseAngleSymmetry import PoseAngleSymmetryData, PoseAngleSymmetryFactory
from modules.pose.Pose import Pose


class PoseSymmetryExtractor(PoseFilterBase):
    """Computes joint angles from pose keypoint data."""

    def process(self, pose: Pose) -> Pose:
        """Compute angles for all poses and emit enriched results."""
        symmetry_data: PoseAngleSymmetryData = PoseAngleSymmetryFactory.from_angles(pose.angle_data)
        return replace(pose, symmetry_data=symmetry_data)