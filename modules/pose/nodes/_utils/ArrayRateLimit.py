# Standard library imports
from typing import Union
from time import monotonic

# Third-party imports
import numpy as np


class RateLimit:
    """Asymmetric rate limiter for vector data.

    Limits the maximum allowed acceleration and deceleration of each vector component.
    Useful for post-processing velocities or other signals to prevent sudden jumps.
    """

    def __init__(self, vector_size: int, max_increase: float , max_decrease: float,
                 clamp_range: tuple[float, float] | None = None) -> None:
        """
        Args:
            vector_size: Number of vector components.
            max_increase: Maximum allowed acceleration (units/s²).
            max_decrease: Maximum allowed deceleration (units/s²).
        """
        if vector_size <= 0:
            raise ValueError("vector_size must be positive")
        if max_increase < 0 or max_decrease < 0:
            raise ValueError("max_increase and max_decrease must be non-negative")

        self._vector_size = vector_size
        self._max_increase = max_increase
        self._max_decrease = max_decrease
        self._clamp_range: tuple[float, float] | None = clamp_range

        self._limited: np.ndarray = np.full(vector_size, np.nan)
        self._target: np.ndarray = np.full(vector_size, np.nan)
        self._last_update_time: float | None = None

    def update(self, values: np.ndarray, current_time: float | None = None) -> None:
        """Apply rate limiting to the new input vector."""
        if values.shape[0] != self._vector_size:
            raise ValueError(f"Expected array of size {self._vector_size}, got {values.shape[0]}")

        self._target = values.copy()

        # HANDLE NAN
        was_nan = np.isnan(self._limited)
        is_nan = np.isnan(self._target)
        state_changed = was_nan != is_nan

        # Where state changed, copy new value and reset velocity
        self._limited[state_changed] = self._target[state_changed]

        # UPDATE
        if current_time is None:
            current_time = monotonic()

        if self._last_update_time is None:
            self._last_update_time = current_time
            return

        dt: float = current_time - self._last_update_time
        self._last_update_time = current_time

        # Only process where both are valid
        valid = ~np.isnan(self._limited) & ~np.isnan(self._target)
        if np.any(valid):
            position_error = self._target[valid] - self._limited[valid]

            # Indices of valid elements
            valid_indices = np.where(valid)[0]

            # Increase: limit by max_increase, Decrease: limit by max_decrease
            increase_mask = position_error > 0
            decrease_mask = position_error < 0

            # Limit increase
            max_inc = self._max_increase * dt
            inc = np.minimum(position_error[increase_mask], max_inc)
            self._limited[valid_indices[increase_mask]] += inc

            # Limit decrease
            max_dec = self._max_decrease * dt
            dec = np.maximum(position_error[decrease_mask], -max_dec)
            self._limited[valid_indices[decrease_mask]] += dec

        # Apply constraints
        self._apply_constraints()

    def _apply_constraints(self) -> None:
        """Apply value constraints (clamping for vectors, overridden for angles)."""
        if self._clamp_range is not None:
            np.clip(self._limited, self._clamp_range[0], self._clamp_range[1], out=self._limited)

    def reset(self) -> None:
        self._limited = np.full(self._vector_size, np.nan)
        self._target = np.full(self._vector_size, np.nan)
        self._velocity = np.zeros(self._vector_size)
        self._last_update_time = None

    @property
    def value(self) -> np.ndarray:
        """Get the current limited value (returns a copy)."""
        return self._limited.copy()

    @property
    def target(self) -> np.ndarray:
        """Get the current target value (returns a copy)."""
        return self._target.copy()

    @ property
    def max_increase(self) -> float:
        """Get the maximum increase rates (returns a copy)."""
        return self._max_increase
    @ max_increase.setter
    def max_increase(self, value: float) -> None:
        """Set the maximum increase rates."""
        self._max_increase = value
        if self._max_increase < 0:
            raise ValueError("max_increase must be non-negative")

    @ property
    def max_decrease(self) -> float:
        """Get the maximum decrease rates (returns a copy)."""
        return self._max_decrease
    @ max_decrease.setter
    def max_decrease(self, value: float) -> None:
        """Set the maximum decrease rates."""
        self._max_decrease = value
        if self._max_decrease < 0:
            raise ValueError("max_decrease must be non-negative")

    @ property
    def clamp_range(self) -> tuple[float, float] | None:
        """Get the clamp range."""
        return self._clamp_range
    @ clamp_range.setter
    def clamp_range(self, value: tuple[float, float] | None) -> None:
        """Set the clamp range."""
        self._clamp_range = value

class AngleRateLimit(RateLimit):
    """Asymmetric rate limiter for angular/circular data with proper wrapping.

    Rate limits are applied to the shortest angular path between current and target values.
    """

    def __init__(self, vector_size: int, max_increase: float, max_decrease: float,
                 clamp_range: tuple[float, float] | None = None) -> None:
        """Initialize the angle rate limiter.

        Args:
            vector_size: Number of angles to limit
            max_increase: Maximum allowed increase per second (radians/s)
            max_decrease: Maximum allowed decrease per second (radians/s)
            clamp_range: Ignored for angles (wrapping is always applied)
        """
        super().__init__(vector_size, max_increase, max_decrease, clamp_range=None)

    def update(self, values: np.ndarray, current_time: float | None = None) -> None:
        """Add new target angles, calculating shortest angular path."""
        if values.shape[0] != self._vector_size:
            raise ValueError(f"Expected array of size {self._vector_size}, got {values.shape[0]}")

        target = values.copy()

        # Only calculate shortest path where both current and new values are valid
        valid = np.isfinite(self._limited) & np.isfinite(values)

        if np.any(valid):
            # Calculate shortest angular path for valid components only
            delta = values[valid] - self._limited[valid]
            delta = np.arctan2(np.sin(delta), np.cos(delta))
            target[valid] = self._limited[valid] + delta

        # For components where _limited is NaN but values is valid,
        # target already contains the raw values (direct initialization)
        super().update(target, current_time)


    def _apply_constraints(self) -> None:
        """Apply angular wrapping to [-π, π]."""
        self._limited = np.arctan2(np.sin(self._limited), np.cos(self._limited))


class PointRateLimit(RateLimit):
    """Asymmetric rate limiter for 2D points with (x, y) coordinates."""

    def __init__(self, vector_size: int, max_increase: float, max_decrease: float,
                 clamp_range: tuple[float, float] | None = None) -> None:
        """Initialize the point rate limiter.

        Args:
            vector_size: Number of points to limit
            max_increase: Maximum allowed increase per second (units/s)
            max_decrease: Maximum allowed decrease per second (units/s)
            clamp_range: Optional (min, max) tuple to clamp point coordinates
        """
        super().__init__(vector_size * 2, max_increase, max_decrease, clamp_range)
        self._num_points: int = vector_size

    def update(self, points: np.ndarray, current_time: float | None = None) -> None:
        """Add new target points."""
        if points.shape != (self._num_points, 2):
            raise ValueError(f"Expected shape ({self._num_points}, 2), got {points.shape}")
        super().update(points.copy().flatten() , current_time)

    @property
    def value(self) -> np.ndarray:
        """Get the current limited points (returns a copy)."""
        return self._limited.reshape(self._num_points, 2).copy()

ArrayRateLimit = Union[RateLimit, AngleRateLimit, PointRateLimit]