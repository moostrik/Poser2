"""Ring buffer for accumulating feature windows from pose frames."""

from enum import IntEnum
from threading import Lock
import numpy as np

from modules.pose.frame import Frame
from modules.pose.frame.window import FeatureWindow
from modules.pose.features import BaseScalarFeature
from modules.settings import BaseSettings, Field


class WindowNodeSettings(BaseSettings):
    """Configuration for window nodes."""
    window_size:  Field[int]  = Field(30)
    emit_partial: Field[bool] = Field(True)


class WindowNode:
    """Buffers pose frames and outputs feature windows.

    Uses a preallocated numpy ring buffer for performance.
    Always returns full window_size buffer with oldest samples first.
    Unfilled slots have mask=False to indicate no data is present.
    Thread-safe: config changes and buffer operations are protected by lock.

    Output:
        FeatureWindow with values and mask arrays, both shape (window_size, feature_len), oldest first.
        Unfilled slots have mask=False.
    """

    def __init__(self, feature_type: type[BaseScalarFeature], config: WindowNodeSettings | None = None) -> None:
        self._feature_type: type[BaseScalarFeature] = feature_type
        self._config: WindowNodeSettings = config if config is not None else WindowNodeSettings()
        self._lock: Lock = Lock()
        self._head: int = 0  # Next insertion index
        self._count: int = 0  # Number of filled slots

        # Preallocate buffers
        self._allocate_buffers()

        # Listen for config changes
        self._config.bind_all(self._on_config_change)

    @property
    def feature_type(self) -> type[BaseScalarFeature]:
        """Return the feature type this window node extracts."""
        return self._feature_type

    @property
    def feature_enum(self) -> type[IntEnum]:
        """Return the feature enum class, derived from feature type."""
        return self._feature_type.enum()

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
        feature = frame[self._feature_type]

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
        if self._head == 0:
            values = self._values.copy()
            mask = self._mask.copy()
        else:
            # Slice concatenation - avoids roll overhead
            values = np.concatenate([self._values[self._head:], self._values[:self._head]], axis=0)
            mask = np.concatenate([self._mask[self._head:], self._mask[:self._head]], axis=0)

        return FeatureWindow(
            values=values,
            mask=mask,
            feature_enum=self.feature_enum,
            range=self._feature_type.range(),
            display_range=self._feature_type.display_range()
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

    def _on_config_change(self, _=None) -> None:
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
    def config(self) -> WindowNodeSettings:
        """Access the node configuration."""
        return self._config
