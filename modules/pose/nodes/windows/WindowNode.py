from abc import abstractmethod
from dataclasses import dataclass
from enum import IntEnum
from threading import Lock
from typing import Generic, TypeVar

import numpy as np

from modules.pose.Frame import Frame, FrameField
from modules.pose.features import BaseScalarFeature, Angles, AngleMotion, AngleSymmetry, AngleVelocity, BBox, Similarity
from modules.pose.nodes.Nodes import NodeBase, NodeConfigBase


TEnum = TypeVar('TEnum', bound=IntEnum)


@dataclass(frozen=True)
class FeatureWindow(Generic[TEnum]):
    """Window of feature values with metadata.

    Shape: (time, feature_len) where time is oldest-first.

    Attributes:
        values: Feature values array of shape (time, feature_len)
        mask: Boolean validity mask of shape (time, feature_len)
        feature_enum: Enum class for feature elements (e.g., AngleLandmark)
        range: (min, max) tuple for validation range
        display_range: (min, max) tuple for visualization range
    """
    values: np.ndarray       # (time, feature_len)
    mask: np.ndarray         # (time, feature_len) bool
    feature_enum: type[TEnum]  # e.g., AngleLandmark
    range: tuple[float, float]  # (min, max) validation range
    display_range: tuple[float, float]  # (min, max) visualization range

    @property
    def feature_names(self) -> list[str]:
        """Derive feature names from enum."""
        return [e.name for e in self.feature_enum]

    def __getitem__(self, landmark: TEnum | int) -> np.ndarray:
        """Get time series for one feature element.

        Example: window[AngleLandmark.left_elbow] returns (time,) array
        """
        return self.values[:, landmark]

    @property
    def shape(self) -> tuple[int, ...]:
        """Return (time, feature_len) shape."""
        return self.values.shape

    @property
    def time_len(self) -> int:
        """Number of time samples in window."""
        return self.values.shape[0]

    @property
    def feature_len(self) -> int:
        """Number of feature elements."""
        return self.values.shape[1]


class WindowNodeConfig(NodeConfigBase):
    """Configuration for window nodes."""

    def __init__(self, window_size: int = 30, emit_partial: bool = True) -> None:
        super().__init__()
        self.window_size = window_size
        self.emit_partial = emit_partial  # If False, only emit when window is full


TFeature = TypeVar('TFeature', bound=BaseScalarFeature)


class WindowNode(NodeBase, Generic[TFeature]):
    """Base class for nodes that buffer poses and output feature windows.

    Uses a preallocated numpy ring buffer for performance.
    Always returns full window_size buffer with oldest samples first.
    Unfilled slots have mask=False to indicate no data is present.
    Thread-safe: config changes and buffer operations are protected by lock.

    Output:
        FeatureWindow with values and mask arrays, both shape (window_size, feature_len), oldest first.
        Unfilled slots have mask=False.
    """

    def __init__(self, frame_field: FrameField, config: WindowNodeConfig | None = None) -> None:
        self._frame_field = frame_field
        self._config: WindowNodeConfig = config if config is not None else WindowNodeConfig()
        self._lock: Lock = Lock()
        self._head: int = 0  # Next insertion index
        self._count: int = 0  # Number of filled slots

        # Preallocate buffers
        self._allocate_buffers()

        # Listen for config changes
        self._config.add_listener(self._on_config_change)

    @property
    def frame_field(self) -> FrameField:
        """Return the FrameField this window node extracts."""
        return self._frame_field

    @property
    def feature_enum(self) -> type[IntEnum]:
        """Return the feature enum class, derived from feature type."""
        feature_type = self.frame_field.get_type()
        return feature_type.enum()

    @property
    def feature_length(self) -> int:
        """Return the feature dimension, derived from enum."""
        return len(self.feature_enum)

    def _allocate_buffers(self) -> None:
        """Allocate numpy buffers based on config."""
        window_size = self._config.window_size
        feature_len = self.feature_length
        self._values = np.zeros((window_size, feature_len), dtype=np.float32)
        self._mask = np.zeros((window_size, feature_len), dtype=bool)

    def process(self, frame: Frame) -> FeatureWindow | None:
        """Buffer frame's feature and return current window state.

        Returns:
            FeatureWindow with values and mask arrays, both shape (current_len, feature_len), oldest first.
            Returns None if emit_partial=False and window not yet full.
        """
        # Extract feature from frame
        feature: TFeature = frame.get_feature(self.frame_field)

        with self._lock:
            # Write to ring buffer at head position
            self._values[self._head] = np.nan_to_num(feature.values, nan=0.0)
            self._mask[self._head] = feature.valid_mask

            # Advance head
            self._head = (self._head + 1) % self._config.window_size

            # Track fill count
            if self._count < self._config.window_size:
                self._count += 1

            # Check if we should emit
            if not self._config.emit_partial and self._count < self._config.window_size:
                return None

            # Return reordered view (oldest first)
            return self._get_ordered_window()

    def _get_ordered_window(self) -> FeatureWindow:
        """Return full-sized window with oldest sample at index 0.
        
        Always returns window_size buffer. Unfilled slots have mask=False.
        """
        # Always use np.roll to reorder so oldest is first
        # _head points to next insert = oldest slot
        values = np.roll(self._values, -self._head, axis=0).copy()
        mask = np.roll(self._mask, -self._head, axis=0).copy()

        return FeatureWindow(
            values=values,
            mask=mask,
            feature_enum=self.feature_enum,
            range=self.frame_field.get_range(),
            display_range=self.frame_field.get_display_range()
        )

    def _reset_buffer(self) -> None:
        """Reset buffer state without reallocating."""
        self._head = 0
        self._count = 0
        self._values.fill(0)
        self._mask.fill(False)

    def reset(self) -> None:
        """Clear the buffer."""
        with self._lock:
            self._reset_buffer()

    def is_ready(self) -> bool:
        """Window nodes are always ready to process."""
        return True

    def _on_config_change(self) -> None:
        """Handle config changes - reallocate buffer if size changed."""
        with self._lock:
            if self._values.shape[0] != self._config.window_size:
                self._allocate_buffers()
                self._reset_buffer()

    @property
    def current_size(self) -> int:
        """Current number of frames in buffer."""
        return self._count

    @property
    def config(self) -> WindowNodeConfig:
        """Access the node configuration."""
        return self._config


    # CONVIENCE CLASSES
def AngleWindowNode(config: WindowNodeConfig | None = None) -> WindowNode[Angles]:
    """Angle trajectories - shape (time, 9) for 9 angles."""
    return WindowNode(FrameField.angles, config)

def AngleVelocityWindowNode(config: WindowNodeConfig | None = None) -> WindowNode[AngleVelocity]:
    """Angular velocity trajectories - shape (time, 9)."""
    return WindowNode(FrameField.angle_vel, config)

def AngleMotionWindowNode(config: WindowNodeConfig | None = None) -> WindowNode[AngleMotion]:
    """Angular motion magnitude - shape (time, 9)."""
    return WindowNode(FrameField.angle_motion, config)

def AngleSymmetryWindowNode(config: WindowNodeConfig | None = None) -> WindowNode[AngleSymmetry]:
    """Angular symmetry metrics - shape (time, 9)."""
    return WindowNode(FrameField.angle_sym, config)

def BBoxWindowNode(config: WindowNodeConfig | None = None) -> WindowNode[BBox]:
    """Bounding box trajectory - shape (time, 4) for [x, y, w, h]."""
    return WindowNode(FrameField.bbox, config)

def SimilarityWindowNode(config: WindowNodeConfig | None = None) -> WindowNode[Similarity]:
    """Similarity trajectory - shape (time, 1)."""
    return WindowNode(FrameField.similarity, config)