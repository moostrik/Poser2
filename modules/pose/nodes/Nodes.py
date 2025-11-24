from abc import ABC, abstractmethod
from typing import Any, Callable, Generic, TypeVar
from modules.pose.Pose import Pose


class NodeConfigBase:
    """Base class for node configurations with automatic change notification and parameter validation."""

    # Subclasses define parameter ranges as class attributes
    _PARAM_RANGES: dict[str, tuple[float | None, float | None]] = {}

    def __init__(self) -> None:
        self._listeners: list[Callable[[], None]] = []

    def __setattr__(self, name: str, value: Any) -> None:
        """Intercept attribute changes to clamp values and notify listeners."""
        # Don't process private attributes or _listeners itself
        if not name.startswith('_'):
            # Clamp numeric values to defined ranges
            if name in self._PARAM_RANGES and isinstance(value, (int, float)):
                min_val, max_val = self._PARAM_RANGES[name]
                if min_val is not None:
                    value = max(min_val, value)
                if max_val is not None:
                    value = min(max_val, value)

        # Set the value
        super().__setattr__(name, value)

        # Notify listeners after initialization
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

    def get_param_range(self, param_name: str) -> tuple[float | None, float | None]:
        """Get the valid range for a parameter."""
        return self._PARAM_RANGES.get(param_name, (None, None))

    def get_all_param_ranges(self) -> dict[str, tuple[float | None, float | None]]:
        """Get ranges for all parameters in this config."""
        return self._PARAM_RANGES.copy()


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
    def process(self, pose: Pose) -> Pose:
        """Process a pose and return the result immediately."""
        pass

    def reset(self) -> None:
        """most filters have no internal state to reset."""
        pass

    def is_ready(self) -> bool:
        """Filters are always ready."""
        return True


class InterpolatorNode(NodeBase):
    """Base class for interpolator nodes that smooth/blend poses."""

    @abstractmethod
    def submit(self, pose: Pose | None) -> None:
        """Set interpolation target from input pose. Called at input frequency (~30 FPS)."""
        pass

    @abstractmethod
    def update(self, time_stamp: float | None = None) -> Pose | None:
        """Get interpolated pose. Called at render frequency (~60+ FPS). Returns None if not ready."""
        pass

    @property
    @abstractmethod
    def attr_name(self) -> str:
        """Return the attribute name this interpolator processes, or None if not applicable."""
        pass


class GeneratorNode(NodeBase, Generic[TInput]):
    """Base class for nodes that generate Pose objects from external data sources."""

    @abstractmethod
    def submit(self, input_data: TInput | None) -> None:
        """Set the input data for pose generation."""
        pass

    @abstractmethod
    def update(self, time_stamp: float | None = None) -> Pose:
        """Generate a new pose."""
        pass


class ProcessorNode(NodeBase, Generic[TInput, TOutput]):
    """Base class for processor nodes that extract derived data from poses using stored context."""

    @abstractmethod
    def submit(self, input_data: TInput | None) -> None:
        """Set the context data for processing."""
        pass

    @abstractmethod
    def process(self, pose: Pose) -> tuple[Pose, TOutput]: # see if we can remove Pose from output
        """Process pose using stored context to produce derived output."""
        pass