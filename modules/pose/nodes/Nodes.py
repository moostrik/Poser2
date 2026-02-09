from abc import ABC, abstractmethod
from typing import Any, Callable, Generic, TypeVar
from modules.pose.Frame import Frame, FrameField


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


TInput = TypeVar('TInput')
TOutput = TypeVar('TOutput')


class NodeBase(ABC):
    """Abstract base class for pose processing nodes."""

    @abstractmethod
    def reset(self) -> None:
        """Optional reset the node's internal state."""
        pass

    @abstractmethod
    def is_ready(self) -> bool:
        """Check if node is ready to process. Override if node has prerequisites."""
        pass


class FilterNode(NodeBase):
    """Base class for filter extractor nodes that modify/transform poses."""

    @abstractmethod
    def process(self, pose: Frame) -> Frame:
        """Process a pose and return the result immediately."""
        pass

    def submit(self, context: Any) -> None:
        """Optional method to submit external context data. Override if filter needs context."""
        pass

    def reset(self) -> None:
        """most filters have no internal state to reset."""
        pass

    def is_ready(self) -> bool:
        """most filters are always ready."""
        return True


class InterpolatorNode(NodeBase):
    """Base class for interpolator nodes that smooth/blend poses."""

    @abstractmethod
    def submit(self, pose: Frame | None) -> None:
        """Set interpolation target from input pose. Called at input frequency (~30 FPS)."""
        pass

    @abstractmethod
    def update(self) -> Frame | None:
        """Get interpolated pose. Called at render frequency (~60+ FPS). Returns None if not ready."""
        pass

    @property
    @abstractmethod
    def pose_field(self) -> FrameField:
        """Return the FrameField this interpolator processes."""
        pass


class ProcessorNode(NodeBase, Generic[TInput, TOutput]):
    """Base class for processor nodes that extract derived data from poses using stored context."""

    @abstractmethod
    def submit(self, input_data: TInput | None) -> None:
        """Set the context data for processing."""
        pass

    @abstractmethod
    def process(self, pose: Frame) -> tuple[Frame, TOutput]:
        """Process pose using stored context to produce derived output."""
        pass