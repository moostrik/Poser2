

from modules.pose.Pose import Pose
from modules.pose.Nodes import FilterNode
from modules.pose.callback import PoseCallbackMixin


class CallbackFilter(PoseCallbackMixin):
    """
    Wraps a PoseFilterBase and provides callback registration for push-based filters.

    Compatible with:
    - All filters derived from PoseFilterBase that operate in a push-based manner.

    NOT compatible with:
    - PoseChaseInterpolator (uses pull-based architecture)

    Usage:
        filter = PoseSmoother(config)
        callback_filter = PoseCallbackFilter(filter)
        callback_filter.add_callback(my_callback)
        callback_filter.add_pose(pose)  # Triggers callback immediately
    """

    def __init__(self, filter_instance: FilterNode) -> None:
        self._filter: FilterNode = filter_instance

    def add_pose(self, pose: Pose) -> None:
        """Process a pose and notify all registered callbacks."""
        processed_pose: Pose = self._filter.process(pose)
        self._notify_callbacks(processed_pose)

        if pose.lost:
            self.reset()

    def reset(self) -> None:
        """Reset the filter's internal state."""
        self._filter.reset()