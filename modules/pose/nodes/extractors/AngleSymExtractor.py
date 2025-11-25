# Standard library imports
from dataclasses import replace

# Pose imports
from modules.pose.nodes.Nodes import FilterNode
from modules.pose.features import AngleSymmetry
from modules.pose.nodes._utils.SymmetryUtils import SymmetryUtils
from modules.pose.Frame import Frame


class AngleSymExtractor(FilterNode):
    """Computes joint angles from pose keypoint data."""

    def process(self, pose: Frame) -> Frame:
        """Compute angles for all poses and emit enriched results."""
        angle_sym: AngleSymmetry = SymmetryUtils.from_angles(pose.angles, 1.0) # this can be parameterized later
        return replace(pose, angle_sym=angle_sym)