# Pose imports
from modules.pose.features import Angles, Points2D
from modules.pose.nodes.Nodes import FilterNode
from modules.pose.nodes._utils.AngleUtils import AngleUtils
from modules.pose.frame import Frame, replace
from modules.settings import Settings, Field


class AngleExtractorSettings(Settings):
    """Configuration for AngleExtractor."""
    aspect_ratio: Field[float] = Field(0.75, access=Field.INIT)


class AngleExtractor(FilterNode):
    """Computes joint angles from pose keypoint data.

    Calculates:
    - Joint angles: Angular measurements at body joints (shoulders, elbows, hips, knees)
    - Head yaw: Head rotation relative to torso

    Uses PoseAngleFactory to compute angles from 2D keypoint positions with proper
    rotation offsets and symmetric mirroring for right-side joints.
    """

    def __init__(self, config: AngleExtractorSettings | None = None) -> None:
        self._config = config if config is not None else AngleExtractorSettings()

    def process(self, pose: Frame) -> Frame:
        """Compute angles for all poses and emit enriched results."""
        angles: Angles = AngleUtils.from_points(pose[Points2D], aspect_ratio=self._config.aspect_ratio)
        return replace(pose, {Angles: angles})