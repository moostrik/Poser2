

from modules.pose.Pose import Pose, PoseCallback
from modules.pose.filters.PoseFilterBase import PoseFilterBase
from modules.pose.callbacks import PoseCallbackMixin
from threading import Lock


class PoseCallbackFilter(PoseCallbackMixin):
    """
    Wraps a PoseFilterBase and provides callback registration for push-based filters.

    Compatible with:
    - PoseSmoother, PoseAngleSmoother, PosePointSmoother, PoseDeltaSmoother
    - PosePredictor, PoseAnglePredictor, PosePointPredictor, PoseDeltaPredictor
    - Other filters that process synchronously

    NOT compatible with:
    - PoseChaseInterpolator and its variants (use pull-based architecture instead)

    Usage:
        filter = PoseSmoother(config)
        callback_filter = PoseCallbackFilter(filter)
        callback_filter.add_callback(my_callback)
        callback_filter.add_pose(pose)  # Triggers callback immediately
    """

    def __init__(self, filter_instance: PoseFilterBase) -> None:
        self._filter: PoseFilterBase = filter_instance

    def add_pose(self, pose: Pose) -> None:
        """Process a pose and notify all registered callbacks."""
        processed_pose: Pose = self._filter.process(pose)
        self._notify_callbacks(processed_pose)

        if pose.lost:
            self.reset()

    def reset(self) -> None:
        """Reset the filter's internal state."""
        self._filter.reset()