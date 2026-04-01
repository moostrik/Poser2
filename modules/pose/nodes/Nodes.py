from abc import ABC, abstractmethod
from typing import Any
from modules.pose.Frame import Frame, FrameField


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
