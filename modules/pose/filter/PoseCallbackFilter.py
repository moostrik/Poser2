from modules.pose.Pose import Pose, PoseCallback
from modules.pose.filter.PoseFilterBase import PoseFilterBase
from threading import Lock

class PoseCallbackFilter:
    """
    Wraps a PoseFilterBase and provides callback registration.
    Allows single-pose filters to be used in a callback-driven style.
    """

    def __init__(self, filter_instance: PoseFilterBase) -> None:
        self._filter: PoseFilterBase = filter_instance
        self._callbacks: set[PoseCallback] = set()
        self._callback_lock = Lock()

    def add_pose(self, pose: Pose) -> None:
        """Process a pose and notify all callbacks with the result."""
        self._notify_callbacks(self._filter.process(pose))

        if pose.lost:
            self.reset()

    def reset(self) -> None:
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