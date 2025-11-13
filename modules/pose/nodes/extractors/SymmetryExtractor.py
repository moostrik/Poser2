# Standard library imports
from dataclasses import replace

# Pose imports
from modules.pose.nodes.Nodes import FilterNode
from modules.pose.features import SymmetryFeature
from modules.pose.nodes._utils.SymmetryUtils import SymmetryUtils
from modules.pose.Pose import Pose


class SymmetryExtractor(FilterNode):
    """Computes joint angles from pose keypoint data."""

    def process(self, pose: Pose) -> Pose:
        """Compute angles for all poses and emit enriched results."""
        symmetry: SymmetryFeature = SymmetryUtils.from_angles(pose.angles, 1.0) # this can be parameterized later
        return replace(pose, symmetry=symmetry)