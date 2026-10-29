"""
Linear interpolation for vector data based on input frequency.

This interpolator performs true linear interpolation between old and new target values
over exactly one input interval. When a new target arrives, it linearly blends from the
current interpolated position to the new target over the time period of one input cycle.

Unlike chase interpolators, this has no velocity or damping - just pure linear movement
from point A to point B over a fixed time period determined by the input frequency.

Note: These classes are NOT thread-safe by design.
"""

# Standard library imports
from typing import Union

# Third-party imports
import numpy as np


class Lerp:
    """Linear interpolator for arbitrary vector data (positions, coordinates, etc.)."""

    def __init__(self, vector_size: int, input_frequency: float = 30.0,
                 clamp_range: tuple[float, float] | None = None) -> None:
        """Initialize the linear interpolator.

        This interpolator performs true linear interpolation between values over exactly
        one input interval. When set_target() is called with new values, it begins linearly
        interpolating from the current position to the new target, completing the movement
        over one input period (1 / input_frequency seconds).

        Example with input_frequency = 30.0 (30 FPS):
        - t=0.000s: set_target([10]) starts interpolation from current position [0]
        - t=0.0167s: halfway through interval, value is [5]
        - t=0.033s: end of interval, value reaches [10]
        - t=0.033s: set_target([20]) starts new interpolation from [10] to [20]

        Args:
            vector_size: Number of values to interpolate simultaneously
            input_frequency: Expected rate of set_target() calls in Hz. Determines how
                           long each interpolation takes (default: 30.0)
            clamp_range: Optional (min, max) tuple to clamp interpolated values

        Raises:
            ValueError: If vector_size <= 0, input_frequency <= 0, or parameters out of range
        """
        if vector_size <= 0:
            raise ValueError(f"vector_size must be positive, got {vector_size}")
        if input_frequency <= 0.0:
            raise ValueError(f"input_frequency must be positive, got {input_frequency}")
        if clamp_range is not None:
            if len(clamp_range) != 2 or clamp_range[0] >= clamp_range[1]:
                raise ValueError("clamp_range must be (min, max) with min < max")

        self._vector_size: int = vector_size
        self._input_interval: float = 1.0 / input_frequency
        self._clamp_range: tuple[float, float] | None = clamp_range

        self._elapsed: float = 0.0

        self._start_values: np.ndarray = np.full(vector_size, np.nan)
        self._target: np.ndarray = np.full(vector_size, np.nan)
        self._interpolated: np.ndarray = np.full(vector_size, np.nan)

    def reset(self) -> None:
        """Reset the interpolator's internal state."""
        self._elapsed = 0.0
        self._start_values = np.full(self._vector_size, np.nan)
        self._target = np.full(self._vector_size, np.nan)
        self._interpolated = np.full(self._vector_size, np.nan)

    @property
    def input_frequency(self) -> float:
        """Get the input frequency (Hz) used for interpolation duration."""
        return 1.0 / self._input_interval

    @input_frequency.setter
    def input_frequency(self, value: float) -> None:
        """Set the input frequency. This determines how long each interpolation takes."""
        if value <= 0.0:
            raise ValueError("Frequency must be positive.")
        self._input_interval = 1.0 / value

    @property
    def clamp_range(self) -> tuple[float, float] | None:
        """Get the clamping range."""
        return self._clamp_range

    @clamp_range.setter
    def clamp_range(self, value: tuple[float, float] | None) -> None:
        """Set the clamping range."""
        if value is not None:
            if len(value) != 2 or value[0] >= value[1]:
                raise ValueError("clamp_range must be (min, max) with min < max")
        self._clamp_range = value

    @property
    def value(self) -> np.ndarray:
        """Get the current interpolated values."""
        return self._interpolated.copy()

    def set_target(self, values: np.ndarray) -> None:
        """Set new target values to interpolate towards.

        This starts a new linear interpolation from the current interpolated position
        to the new target values, taking exactly one input interval to complete.

        Args:
            values: Target values array of size vector_size
        """
        if values.shape[0] != self._vector_size:
            raise ValueError(f"Expected array of size {self._vector_size}, got {values.shape[0]}")

        # Store current interpolated values as the start of the new lerp
        self._start_values = self._interpolated.copy()
        self._target = values.copy()

        # Initialize newly valid values (first time setup)
        newly_valid = np.isnan(self._interpolated) & np.isfinite(values)
        if np.any(newly_valid):
            self._interpolated[newly_valid] = values[newly_valid]
            self._start_values[newly_valid] = values[newly_valid]

        # Reset elapsed time for new lerp
        self._elapsed = 0.0

    def update(self, dt: float) -> None:
        """Update the interpolated value by a time delta.

        Performs linear interpolation: interpolated = start + (target - start) * t
        where t is the progress through the input interval [0.0, 1.0].

        Args:
            dt: Time delta in seconds since last update (e.g., 1/60 for 60 FPS output).
        """
        if np.all(np.isnan(self._target)):
            return

        if dt <= 0:
            return

        # Accumulate elapsed time and calculate progress
        self._elapsed += dt
        t = min(self._elapsed / self._input_interval, 1.0)  # Clamp to [0, 1]

        # Linear interpolation: lerp(a, b, t) = a + (b - a) * t
        self._interpolated = self._start_values + (self._target - self._start_values) * t

        # Apply constraints
        self._apply_constraints()

    def _apply_constraints(self) -> None:
        """Apply value constraints (clamping for vectors, overridden for angles)."""
        if self._clamp_range is not None:
            np.clip(self._interpolated, self._clamp_range[0], self._clamp_range[1], out=self._interpolated)


class AngleLerp(Lerp):
    """Linear interpolator for angular/circular data with proper wrapping."""

    def __init__(self, vector_size: int, input_frequency: float = 30.0,
                 clamp_range: tuple[float, float] | None = None) -> None:
        """Initialize the angle linear interpolator.

        Args:
            vector_size: Number of angles to interpolate
            input_frequency: Expected rate of set_target() calls in Hz
        """
        super().__init__(vector_size, input_frequency, clamp_range=None)

    def set_target(self, values: np.ndarray) -> None:
        """Set new target angles to interpolate towards.

        Calculates the shortest angular path from current to target using wrapping.

        Args:
            values: Target angle values array of size vector_size (in radians)
        """
        if values.shape[0] != self._vector_size:
            raise ValueError(f"Expected array of size {self._vector_size}, got {values.shape[0]}")

        # Store current interpolated values as the start of the new lerp
        self._start_values = self._interpolated.copy()

        # Calculate shortest angular path to target
        # Wrap the difference to [-π, π] to get shortest rotation
        delta = values - self._interpolated
        delta = np.arctan2(np.sin(delta), np.cos(delta))

        # Target is start + shortest path
        self._target = self._interpolated + delta

        # Initialize newly valid values (first time setup)
        newly_valid = np.isnan(self._interpolated) & np.isfinite(values)
        if np.any(newly_valid):
            self._interpolated[newly_valid] = values[newly_valid]
            self._start_values[newly_valid] = values[newly_valid]
            self._target[newly_valid] = values[newly_valid]

        # Reset elapsed time for new lerp
        self._elapsed = 0.0

    def _apply_constraints(self) -> None:
        """Apply angular wrapping to [-π, π]."""
        self._interpolated = np.arctan2(np.sin(self._interpolated), np.cos(self._interpolated))


class PointLerp(Lerp):
    """Linear interpolator for 2D points with (x, y) coordinates."""

    def __init__(self, vector_size: int, input_frequency: float = 30.0,
                 clamp_range: tuple[float, float] | None = None) -> None:
        """Initialize the point linear interpolator.

        Args:
            vector_size: Number of points to interpolate
            input_frequency: Expected rate of set_target() calls in Hz
            clamp_range: Optional clamping range for both x and y coordinates
        """
        super().__init__(vector_size * 2, input_frequency, clamp_range)
        self._num_points: int = vector_size

    def set_target(self, points: np.ndarray) -> None:
        """Set new target points.

        Args:
            points: Array of shape (num_points, 2) with (x, y) coordinates
        """
        if points.shape != (self._num_points, 2):
            raise ValueError(f"Expected shape ({self._num_points}, 2), got {points.shape}")
        super().set_target(points.copy().flatten())

    @property
    def value(self) -> np.ndarray:
        """Get the current interpolated points as (num_points, 2) array."""
        return self._interpolated.reshape(self._num_points, 2).copy()


ArrayLerp = Union[Lerp, AngleLerp, PointLerp]