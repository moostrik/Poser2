# Standard library imports
from dataclasses import replace

# Pose imports
from modules.pose.features import Angles
from modules.pose.nodes.Nodes import FilterNode
from modules.pose.nodes._utils.AngleUtils import AngleUtils
from modules.pose.Frame import Frame


class AngleExtractor(FilterNode):
    """Computes joint angles from pose keypoint data.

    Calculates:
    - Joint angles: Angular measurements at body joints (shoulders, elbows, hips, knees)
    - Head yaw: Head rotation relative to torso

    Uses PoseAngleFactory to compute angles from 2D keypoint positions with proper
    rotation offsets and symmetric mirroring for right-side joints.
    """

    def process(self, pose: Frame) -> Frame:
        """Compute angles for all poses and emit enriched results."""
        angles: Angles = AngleUtils.from_points(pose.points, aspect_ratio=pose.model_ar)
        return replace(pose, angles=angles)