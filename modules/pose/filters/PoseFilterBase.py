# Standard library imports
from abc import ABC, abstractmethod
from threading import Lock
from traceback import print_exc

# Pose imports
from ..Pose import PoseDict, PoseDictCallback


class PoseFilterBase(ABC):
    """Abstract base class for pose pipeline filters.

    Filters receive pose data, process/enrich it, and emit results to registered callbacks.
    Common use cases:
    - Feature computation (angles, velocities, deltas)
    - Data filtering (confidence thresholds, smoothing)
    - State tracking (temporal features, motion analysis)

    Type Parameters:
        TSettings: Type of settings object used by this filter (optional)
    """

    def __init__(self) -> None:
        self.pose_output_callbacks: set[PoseDictCallback] = set()
        self._callback_lock = Lock()

    @abstractmethod
    def add_poses(self, poses: PoseDict) -> None:
        """Process incoming poses and emit results.

        Args:
            poses: Dictionary of pose_id -> Pose to process

        Implementations should:
        1. Process/enrich the poses
        2. Call _notify_callbacks() with results
        """
        pass

    def add_callback(self, callback: PoseDictCallback) -> None:
        """Register a callback to receive processed poses.

        Args:
            callback: Function that accepts PoseDict
        """
        with self._callback_lock:
            self.pose_output_callbacks.add(callback)

    def remove_callback(self, callback: PoseDictCallback) -> None:
        """Unregister a callback.

        Args:
            callback: Previously registered callback function
        """
        with self._callback_lock:
            self.pose_output_callbacks.discard(callback)

    def _notify_callbacks(self, poses: PoseDict) -> None:
        """Notify all registered callbacks with processed poses.

        Args:
            poses: Processed pose dictionary to send to callbacks
        """
        with self._callback_lock:
            for callback in self.pose_output_callbacks:
                try:
                    callback(poses)
                except Exception as e:
                    # Log error but continue notifying other callbacks
                    print(f"{self.__class__.__name__}: Error in callback: {e}")
                    print_exc()