# Standard library imports
from typing import Union
from time import monotonic

# Third-party imports
import numpy as np

class DerivativeRateLimit:
    """Asymmetric rate limiter for the derivative (jerk limiter for velocity).

    Limits the maximum allowed increase and decrease of the rate of change per step.
    For velocity inputs, this acts as a jerk limiter.
    """

    def __init__(self, vector_size: int, max_increase: float | np.ndarray, max_decrease: float | np.ndarray,
                 clamp_range: tuple[float, float] | None = None) -> None:
        """
        Args:
            vector_size: Number of vector components.
            max_increase: Maximum allowed derivative increase per second (units/s²).
            max_decrease: Maximum allowed derivative decrease per second (units/s²).
            clamp_range: Optional (min, max) tuple to clamp output values.
        """
        if vector_size <= 0:
            raise ValueError("vector_size must be positive")
        self._vector_size = vector_size

        # Broadcast max_increase and max_decrease to arrays
        self._max_increase = np.broadcast_to(max_increase, (vector_size,))
        self._max_decrease = np.broadcast_to(max_decrease, (vector_size,))

        if np.any(self._max_increase < 0) or np.any(self._max_decrease < 0):
            raise ValueError("max_increase and max_decrease must be non-negative")

        self._clamp_range: tuple[float, float] | None = clamp_range

        self._limited: np.ndarray = np.full(vector_size, np.nan)  # Current limited derivative
        self._target: np.ndarray = np.full(vector_size, np.nan)   # Target derivative
        self._last_update_time: float | None = None

    def set_target(self, values: np.ndarray) -> None:
        """Set the target derivative values."""
        if values.shape[0] != self._vector_size:
            raise ValueError(f"Expected array of size {self._vector_size}, got {values.shape[0]}")

        self._target = values.copy()

        was_nan = np.isnan(self._limited)
        is_nan = np.isnan(self._target)
        state_changed = was_nan != is_nan

        # Where state changed, copy new value (handles NaN <-> valid)
        self._limited[state_changed] = self._target[state_changed]

    def update(self, current_time: float | None = None) -> None:
        """Update the limited derivative based on elapsed time."""
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
            delta = self._target[valid] - self._limited[valid]
            clamped_delta = np.where(
                delta > 0,
                np.minimum(delta, self._max_increase[valid] * dt),
                np.maximum(delta, -self._max_decrease[valid] * dt),
            )
            self._limited[valid] += clamped_delta

        # Apply constraints
        self._apply_constraints()

    def _apply_constraints(self) -> None:
        """Apply value constraints (clamping)."""
        if self._clamp_range is not None:
            np.clip(self._limited, self._clamp_range[0], self._clamp_range[1], out=self._limited)

    def reset(self) -> None:
        """Reset the limiter state."""
        self._limited = np.full(self._vector_size, np.nan)
        self._target = np.full(self._vector_size, np.nan)
        self._last_update_time = None

    @property
    def value(self) -> np.ndarray:
        """Get the current limited derivative value (returns a copy)."""
        return self._limited.copy()

    @property
    def target(self) -> np.ndarray:
        """Get the current target derivative value (returns a copy)."""
        return self._target.copy()

    @property
    def max_increase(self) -> np.ndarray:
        """Get the maximum increase rates (returns a copy)."""
        return self._max_increase.copy()
    @max_increase.setter
    def max_increase(self, value: float | np.ndarray) -> None:
        """Set the maximum increase rates."""
        self._max_increase = np.broadcast_to(value, (self._vector_size,))
        if np.any(self._max_increase < 0):
            raise ValueError("max_increase must be non-negative")

    @property
    def max_decrease(self) -> np.ndarray:
        """Get the maximum decrease rates (returns a copy)."""
        return self._max_decrease.copy()
    @max_decrease.setter
    def max_decrease(self, value: float | np.ndarray) -> None:
        """Set the maximum decrease rates."""
        self._max_decrease = np.broadcast_to(value, (self._vector_size,))
        if np.any(self._max_decrease < 0):
            raise ValueError("max_decrease must be non-negative")

    @property
    def clamp_range(self) -> tuple[float, float] | None:
        """Get the clamp range."""
        return self._clamp_range
    @clamp_range.setter
    def clamp_range(self, value: tuple[float, float] | None) -> None:
        """Set the clamp range."""
        self._clamp_range = value