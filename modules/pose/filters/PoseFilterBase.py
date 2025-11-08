from abc import ABC, abstractmethod
from threading import Lock
from typing import Any, Callable
from modules.pose.Pose import Pose, PoseCallback


class PoseFilterConfigBase:
    """Base class for filter configurations with automatic change notification."""

    def __init__(self) -> None:
        self._listeners: list[Callable[[], None]] = []

    def __setattr__(self, name: str, value: Any) -> None:
        """Intercept attribute changes and notify listeners."""
        super().__setattr__(name, value)
        # Only notify after initialization is complete
        if name != '_listeners' and hasattr(self, '_listeners'):
            self._notify()

    def add_listener(self, callback: Callable[[], None]) -> None:
        """Register a callback to be notified of config changes."""
        self._listeners.append(callback)

    def remove_listener(self, callback: Callable[[], None]) -> None:
        """Unregister a callback."""
        if callback in self._listeners:
            self._listeners.remove(callback)

    def _notify(self) -> None:
        """Notify all listeners that config has changed."""
        for listener in self._listeners:
            listener()


class PoseFilterBase(ABC):
    """Abstract base class for pose filters."""

    @abstractmethod
    def process(self, pose: Pose) -> Pose:
        """Process the filter"""
        pass

    def reset(self) -> None:
        """Optional reset the filter's internal state."""
        pass
