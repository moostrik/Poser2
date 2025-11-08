from modules.pose.Pose import Pose, PoseCallback
from modules.pose.filters.PoseFilterBase import PoseFilterBase
from threading import Lock


class PoseCallbackFilter:
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
        self._callbacks: set[PoseCallback] = set()
        self._callback_lock = Lock()

    def add_pose(self, pose: Pose) -> None:
        """Process a pose and notify all registered callbacks."""
        processed_pose = self._filter.process(pose)
        self._notify_callbacks(processed_pose)

        if pose.lost:
            self.reset()

    def reset(self) -> None:
        """Reset the filter's internal state."""
        self._filter.reset()

    # CALLBACKS
    def _notify_callbacks(self, pose: Pose) -> None:
        with self._callback_lock:
            for callback in self._callbacks:
                callback(pose)

    def add_callback(self, callback: PoseCallback) -> None:
        with self._callback_lock:
            self._callbacks.add(callback)

    def remove_callback(self, callback: PoseCallback) -> None:
        with self._callback_lock:
            self._callbacks.discard(callback)