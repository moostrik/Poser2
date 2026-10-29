# Pose imports
from ..Nodes import FilterNode
from ...features import Age
from ...frame import Frame, replace


class AgeExtractor(FilterNode):
    """How long this track has had a pose (s): the time since its first frame after a reset."""

    def __init__(self) -> None:
        self.oldest_timestamp: float | None = None


    def process(self, pose: Frame) -> Frame:
        """Compute deltas for all poses and emit enriched results."""

        if self.oldest_timestamp is None:
            self.oldest_timestamp = pose.timestamp
            return pose

        age: float = pose.timestamp - self.oldest_timestamp

        return replace(pose, {Age: Age.from_value(age)})

    def reset(self) -> None:
        self.oldest_timestamp = None