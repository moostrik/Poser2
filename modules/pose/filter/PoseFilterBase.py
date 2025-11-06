from abc import ABC, abstractmethod
from threading import Lock
from modules.pose.Pose import Pose, PoseCallback

class PoseFilterBase(ABC):
    """Abstract base class for pose filters."""

    def __init__(self) -> None:
        self._callbacks: set[PoseCallback] = set()
        self._callback_lock = Lock()

    @abstractmethod
    def process(self, pose: Pose) -> Pose:
        """Process the filter"""
        pass

    def reset(self) -> None:
        """Optional reset the filter's internal state."""
        pass
