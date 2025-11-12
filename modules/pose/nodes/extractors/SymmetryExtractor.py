# Standard library imports
from dataclasses import replace

# Pose imports
from modules.pose.nodes.Nodes import FilterNode
from modules.pose.features import SymmetryFeature, SymmetryFactory
from modules.pose.Pose import Pose


class SymmetryExtractor(FilterNode):
    """Computes joint angles from pose keypoint data."""

    def process(self, pose: Pose) -> Pose:
        """Compute angles for all poses and emit enriched results."""
        symmetry_data: SymmetryFeature = SymmetryFactory.from_angles(pose.angles)
        return replace(pose, symmetry_data=symmetry_data)