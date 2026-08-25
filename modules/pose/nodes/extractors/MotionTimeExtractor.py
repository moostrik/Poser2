# Standard library imports
import math

# Third-party imports
import numpy as np

# Pose imports
from ..Nodes import FilterNode
from ...features import AngleVelocity, MotionTime
from ...frame import Frame, replace


class MotionTimeExtractor(FilterNode):
    """Integrates absolute joint angular velocity over time into a cumulative ``MotionTime``.

    Accumulates only on ticks with a real, positive ``dt`` and ignores non-finite velocity
    components (``inf``/``NaN``). Interpolated frames often repeat a timestamp (``dt == 0``)
    and ``AngleVelocity`` is unbounded, so without these guards an ``inf * 0`` would poison the
    accumulator to ``NaN`` permanently. ``MotionTime`` is stamped every frame — ``0.0`` from the
    first — so the feature is never absent downstream.
    """

    def __init__(self) -> None:
        self.motion_time: float = 0.0
        self.prev_time_stamp: float | None = None

    def process(self, pose: Frame) -> Frame:
        """Accumulate finite angular motion since the previous frame and stamp ``MotionTime``."""

        if self.prev_time_stamp is not None:
            dt: float = pose.time_stamp - self.prev_time_stamp
            if math.isfinite(dt) and dt > 0.0:
                velocities = pose[AngleVelocity].values
                finite = np.isfinite(velocities)              # drop inf and NaN, not just NaN
                total_delta: float = float(np.abs(velocities[finite]).sum()) / np.pi * dt
                if math.isfinite(total_delta):
                    self.motion_time += total_delta
        self.prev_time_stamp = pose.time_stamp

        return replace(pose, {MotionTime: MotionTime.from_value(self.motion_time)})

    def reset(self) -> None:
        self.motion_time = 0.0
        self.prev_time_stamp = None
