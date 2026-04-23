# Pose imports
from ..Nodes import FilterNode
from ...features import Angles, AngleSymmetry
from .._utils.SymmetryUtils import SymmetryUtils
from ...frame import Frame, replace


class AngleSymExtractor(FilterNode):
    """Computes joint angles from pose keypoint data."""

    def process(self, pose: Frame) -> Frame:
        """Compute angles for all poses and emit enriched results."""
        angle_sym: AngleSymmetry = SymmetryUtils.from_angles(pose[Angles], 1.0) # this can be parameterized later
        return replace(pose, {AngleSymmetry: angle_sym})