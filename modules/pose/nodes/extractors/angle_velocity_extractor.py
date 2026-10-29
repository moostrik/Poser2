# Pose imports
from ..nodes import FilterNode
from ...features import Angles, AngleVelocity
from ...frame import Frame, replace
from modules.settings import BaseSettings


class AngleVelocityExtractorSettings(BaseSettings):
    """Configuration for AngleVelocityExtractor; no fields, the time between frames comes from their stamps."""
    pass


class AngleVelocityExtractor(FilterNode):
    """The angular velocity (rad/s) of each angle: the change since the previous frame (wrapped the short way)
    over the time between the two frames' stamps. The first frame, and a frame stamped no later than the
    previous one, gets no velocity; a joint that reappears after being invalid gets NaN."""

    def __init__(self, settings: AngleVelocityExtractorSettings | None = None) -> None:
        super().__init__()
        self._settings = settings if settings is not None else AngleVelocityExtractorSettings()
        self._prev_pose: Frame | None = None

    def process(self, pose: Frame) -> Frame:
        if self._prev_pose is None:
            angle_vel: AngleVelocity = AngleVelocity.create_dummy()
        else:
            dt: float = pose.timestamp - self._prev_pose.timestamp
            angle_displacement: Angles = pose[Angles].subtract(self._prev_pose[Angles])
            if dt > 0:
                angle_vel = AngleVelocity(values=angle_displacement.values / dt, scores=angle_displacement.scores)
            else:
                angle_vel = AngleVelocity.create_dummy()

        self._prev_pose = pose
        return replace(pose, {AngleVelocity: angle_vel})

    def reset(self) -> None:
        self._prev_pose = None
