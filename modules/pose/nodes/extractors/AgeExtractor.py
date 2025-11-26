# Standard library imports
from dataclasses import replace

# Pose imports
from modules.pose.nodes.Nodes import FilterNode
from modules.pose.Frame import Frame


class AgeExtractor(FilterNode):
    """Takes the absolute value of all deltas and adds them to the movement_time field."""

    def __init__(self) -> None:
        self.motion_time: float = 0.0
        self.oldest_time_stamp: float | None = None


    def process(self, pose: Frame) -> Frame:
        """Compute deltas for all poses and emit enriched results."""

        if self.oldest_time_stamp is None:
            self.oldest_time_stamp = pose.time_stamp
            return pose

        age: float = pose.time_stamp - self.oldest_time_stamp

        return replace(pose, age=age)

    def reset(self) -> None:
        self.motion_time = 0.0
        self.prev_time_stamp = None