from abc import ABC, abstractmethod
from threading import Lock
from typing import Any, Callable
from modules.pose.Pose import Pose, PoseCallback


class PoseFilterConfigBase:
    """Base class for filter configurations with automatic change notification."""

    def __init__(self) -> None:
        self._listeners: list[Callable[[], None]] = []

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
    """Abstract base class for pose filters.

    Filters can optionally accept a config object that inherits from FilterConfigBase.
    When config properties change, _on_config_changed() is called automatically.
    """

    def __init__(self, config: PoseFilterConfigBase | None = None) -> None:
        self._callbacks: set[PoseCallback] = set()
        self._callback_lock = Lock()
        self._config: PoseFilterConfigBase | None = config

        # Register for config change notifications
        if config is not None:
            config.add_listener(self._on_config_changed)

    @abstractmethod
    def process(self, pose: Pose) -> Pose:
        """Process the filter"""
        pass

    def reset(self) -> None:
        """Optional reset the filter's internal state."""
        pass

    def _on_config_changed(self) -> None:
        """Called when config changes. Override to apply new config values."""
        pass
