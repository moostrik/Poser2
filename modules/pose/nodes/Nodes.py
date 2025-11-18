from abc import ABC, abstractmethod
from typing import Any, Callable, Generic, TypeVar
from modules.pose.Pose import Pose


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


class FilterNode(NodeBase):
    """Base class for filter extractor nodes that modify/transform poses."""

    @abstractmethod
    def process(self, pose: Pose) -> Pose:
        """Process a pose and return the result immediately."""
        pass

    def reset(self) -> None:
        """Optional reset the node's internal state."""
        pass


class GeneratorNode(NodeBase, Generic[TInput]):
    """Base class for nodes that generate Pose objects from external data sources.

    Converts non-pose data (tracklets, images, templates, etc.) into Pose objects
    using a two-step pattern: set input data, then generate pose.
    """

    @abstractmethod
    def set(self, input_data: TInput) -> None:
        """Set the input data for pose generation."""
        pass

    @abstractmethod
    def generate(self, time_stamp: float | None = None) -> Pose:
        """Generate a new pose."""
        pass

    @abstractmethod
    def is_ready(self) -> bool:
        """Check if the generator is ready to produce a pose."""
        pass

    def reset(self) -> None:
        """Optional reset the generator's internal state."""
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

    @property
    @abstractmethod
    def attr_name(self) -> str:
        """Return the attribute name this interpolator processes, or None if not applicable."""
        pass


class ProcessorNode(NodeBase, Generic[TInput, TOutput]):
    """Base class for processor nodes that extract derived data from poses using stored context.

    Stores data of TInput (i.e. images) via set(), then processes poses to produce
    derived outputs (i.e. cropped images) of type TOutput.
    """

    @abstractmethod
    def set(self, input_data: TInput) -> None:
        """Set the context data for processing."""
        pass

    @abstractmethod
    def process(self, pose: Pose) -> tuple[Pose, TOutput]:
        """Process pose using stored context to produce derived output."""
        pass

    @abstractmethod
    def is_ready(self) -> bool:
        """Check if the processor has context data and is ready."""
        pass

    def reset(self) -> None:
        """Optional reset the processor's internal state."""
        pass