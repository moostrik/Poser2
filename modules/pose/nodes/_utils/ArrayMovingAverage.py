# Standard library imports
from enum import IntEnum
from typing import Optional

# Third-party imports
import numpy as np


class WindowType(IntEnum):
    """Types of window weighting for moving average."""
    UNIFORM = 0      # Equal weights for all samples
    TRIANGULAR = 1   # Linear falloff (newest = highest weight)
    GAUSSIAN = 2     # Bell curve weights
    EXPONENTIAL = 3  # Exponential decay (similar to EMA but bounded)


def generate_weights(window_size: int, window_type: WindowType) -> np.ndarray:
    """Generate normalized weight array for the given window type.

    Args:
        window_size: Number of samples in window
        window_type: Type of weighting to apply

    Returns:
        Normalized weights array where newest sample is at index -1
    """
    if window_size <= 0:
        raise ValueError("window_size must be positive")

    if window_type == WindowType.UNIFORM:
        weights = np.ones(window_size)

    elif window_type == WindowType.TRIANGULAR:
        # Linear ramp: [1, 2, 3, ..., N] - newest has highest weight
        weights = np.arange(1, window_size + 1, dtype=np.float64)

    elif window_type == WindowType.GAUSSIAN:
        # Gaussian bell curve centered at newest sample
        # sigma = window_size / 4 gives good falloff
        sigma = window_size / 4.0
        x = np.arange(window_size, dtype=np.float64)
        # Center at end (newest sample)
        weights = np.exp(-0.5 * ((x - (window_size - 1)) / sigma) ** 2)

    elif window_type == WindowType.EXPONENTIAL:
        # Exponential decay from newest to oldest
        # decay_factor controls how fast it falls off
        decay_factor = 3.0 / window_size  # ~5% remaining at oldest
        x = np.arange(window_size, dtype=np.float64)
        weights = np.exp(-decay_factor * (window_size - 1 - x))

    else:
        weights = np.ones(window_size)

    # Normalize so weights sum to 1
    weights /= weights.sum()
    return weights


class MovingAverage:
    """Moving average smoothing for vector data with configurable window types.

    Stores a circular buffer of the last N samples and computes weighted average.
    Supports multiple weighting schemes: uniform, triangular, gaussian, exponential.
    """

    def __init__(
        self,
        vector_size: int,
        window_size: int = 30,
        window_type: WindowType = WindowType.TRIANGULAR,
        clamp_range: Optional[tuple[float, float]] = None
    ) -> None:
        """
        Args:
            vector_size: Number of vector components per sample
            window_size: Number of samples to keep in buffer
            window_type: Type of weighting to apply
            clamp_range: Optional (min, max) tuple to clamp output values
        """
        if vector_size <= 0:
            raise ValueError("vector_size must be positive")
        if window_size <= 0:
            raise ValueError("window_size must be positive")

        self._vector_size = vector_size
        self._window_size = window_size
        self._window_type = window_type
        self._clamp_range = clamp_range

        # Circular buffer: shape (window_size, vector_size)
        self._buffer: np.ndarray = np.full((window_size, vector_size), np.nan)
        self._buffer_index: int = 0
        self._sample_count: int = 0

        # Pre-compute weights
        self._weights = generate_weights(window_size, window_type)

        # Cached output
        self._value: np.ndarray = np.full(vector_size, np.nan)

    def update(self, values: np.ndarray) -> None:
        """Add a new sample and update the smoothed output."""
        if values.shape[0] != self._vector_size:
            raise ValueError(f"Expected array of size {self._vector_size}, got {values.shape[0]}")

        # Add to circular buffer
        self._buffer[self._buffer_index] = values
        self._buffer_index = (self._buffer_index + 1) % self._window_size
        self._sample_count = min(self._sample_count + 1, self._window_size)

        # Compute weighted average
        self._compute_average()

    def _compute_average(self) -> None:
        """Compute weighted average from buffer."""
        if self._sample_count == 0:
            self._value.fill(np.nan)
            return

        # Get samples in chronological order (oldest to newest)
        if self._sample_count < self._window_size:
            # Buffer not full yet - use only filled portion
            samples = self._buffer[:self._sample_count]
            weights = self._weights[-self._sample_count:]  # Use last N weights
            weights = weights / weights.sum()  # Re-normalize
        else:
            # Full buffer - reorder to chronological
            samples = np.roll(self._buffer, -self._buffer_index, axis=0)
            weights = self._weights

        # Handle NaN values per-element
        for i in range(self._vector_size):
            col = samples[:, i]
            valid_mask = ~np.isnan(col)

            if not np.any(valid_mask):
                self._value[i] = np.nan
            else:
                valid_vals = col[valid_mask]
                valid_weights = weights[valid_mask]
                valid_weights = valid_weights / valid_weights.sum()  # Re-normalize
                self._value[i] = np.sum(valid_vals * valid_weights)

        # Apply constraints
        if self._clamp_range is not None:
            np.clip(self._value, self._clamp_range[0], self._clamp_range[1], out=self._value)

    def reset(self) -> None:
        """Reset the filter to initial state."""
        self._buffer.fill(np.nan)
        self._buffer_index = 0
        self._sample_count = 0
        self._value.fill(np.nan)

    @property
    def value(self) -> np.ndarray:
        """Get the current smoothed value (returns a copy)."""
        return self._value.copy()

    @property
    def window_size(self) -> int:
        """Get the window size."""
        return self._window_size

    @window_size.setter
    def window_size(self, value: int) -> None:
        """Set window size and regenerate weights. Resets buffer."""
        if value <= 0:
            raise ValueError("window_size must be positive")
        if value != self._window_size:
            self._window_size = value
            self._buffer = np.full((value, self._vector_size), np.nan)
            self._weights = generate_weights(value, self._window_type)
            self.reset()

    @property
    def window_type(self) -> WindowType:
        """Get the window type."""
        return self._window_type

    @window_type.setter
    def window_type(self, value: WindowType) -> None:
        """Set window type and regenerate weights."""
        if value != self._window_type:
            self._window_type = value
            self._weights = generate_weights(self._window_size, value)
            # Don't reset buffer - just recompute with new weights
            self._compute_average()

    @property
    def clamp_range(self) -> Optional[tuple[float, float]]:
        """Get the clamp range."""
        return self._clamp_range

    @clamp_range.setter
    def clamp_range(self, value: Optional[tuple[float, float]]) -> None:
        """Set the clamp range."""
        self._clamp_range = value
