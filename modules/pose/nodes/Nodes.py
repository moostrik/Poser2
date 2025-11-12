from abc import ABC, abstractmethod
from typing import Any, Callable
from modules.pose.Pose import Pose, PoseDict


class NodeConfigBase:
    """Base class for node configurations with automatic change notification."""

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


class NodeBase(ABC):
    """Abstract base class for pose processing nodes."""


class FilterNode(NodeBase):
    """Base class for filter extractor nodes that modify/transform poses."""

    @abstractmethod
    def process(self, pose: Pose) -> Pose:
        """Process a pose and return the result immediately."""
        pass

    def reset(self) -> None:
        """Optional reset the node's internal state."""
        pass


class InterpolatorNode(NodeBase):
    """Base class for interpolator nodes that smooth/blend poses."""

    @abstractmethod
    def submit(self, pose: Pose) -> None: # should be submit
        """Set interpolation target from input pose. Called at input frequency (~30 FPS)."""
        pass

    @abstractmethod
    def update(self, current_time: float | None = None) -> Pose | None:
        """Get interpolated pose. Called at render frequency (~60+ FPS). Returns None if not ready."""
        pass

    def reset(self) -> None:
        """Reset the interpolator's internal state (position, velocity, history)."""
        pass


class GeneratorNode(NodeBase):
    """Base class for generator nodes that create poses."""

    @abstractmethod
    def generate(self) -> Pose:
        """Generate a new pose."""
        pass

    def reset(self) -> None:
        """Optional reset the generator's internal state."""
        pass