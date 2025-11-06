# Standard library imports
from dataclasses import replace

# Third-party imports
import numpy as np

# Pose imports
from modules.pose.filter.PoseFilterBase import PoseFilterBase
from modules.pose.Pose import Pose


class PoseMotionTimeAccumulator(PoseFilterBase):
    """Takes the absolute value of all deltas and adds them to the movement_time field."""


    def process(self, pose: Pose) -> Pose:
        """Compute deltas for all poses and emit enriched results."""

        total_delta: float = np.nansum(np.abs(pose.delta_data.values))
        motion_time: float = pose.motion_time + total_delta

        return replace(pose, motion_time=motion_time)