# Standard library imports
from time import monotonic
from typing import Union

# Third-party imports
import numpy as np


class EMASmooth:
    """Exponential Moving Average (EMA) smoothing for vector data with attack/release.

    Provides asymmetric smoothing with different response rates for increasing vs decreasing values.
    Uses exponential smoothing that approaches target asymptotically.
    """

    def __init__(self, vector_size: int, attack: float = 0.5, release: float = 0.2,
                 clamp_range: tuple[float, float] | None = None) -> None:
        """
        Args:
            vector_size: Number of vector components.
            attack: Smoothing factor per second for rising values (0-1). Higher = more responsive.
            release: Smoothing factor per second for falling values (0-1). Higher = more responsive.
            clamp_range: Optional (min, max) tuple to clamp values after smoothing.
        """
        if vector_size <= 0:
            raise ValueError("vector_size must be positive")
        if attack <= 0 or attack > 1.0:
            raise ValueError("attack must be in (0.0, 1.0]")
        if release <= 0 or release > 1.0:
            raise ValueError("release must be in (0.0, 1.0]")

        self._vector_size = vector_size
        self._attack = float(attack)
        self._release = float(release)
        self._clamp_range: tuple[float, float] | None = clamp_range

        self._smoothed: np.ndarray = np.full(vector_size, np.nan)
        self._target: np.ndarray = np.full(vector_size, np.nan)
        self._last_update_time: float | None = None

    def update(self, values: np.ndarray, current_time: float | None = None) -> None:
        """Apply EMA smoothing to the new input vector."""
        if values.shape[0] != self._vector_size:
            raise ValueError(f"Expected array of size {self._vector_size}, got {values.shape[0]}")

        self._target = values.copy()

        # HANDLE NAN
        was_nan = np.isnan(self._smoothed)
        is_nan = np.isnan(self._target)
        state_changed = was_nan != is_nan

        # Where state changed, copy new value
        self._smoothed[state_changed] = self._target[state_changed]

        # UPDATE
        if current_time is None:
            current_time = monotonic()

        if self._last_update_time is None:
            self._last_update_time = current_time
            return

        dt: float = current_time - self._last_update_time
        self._last_update_time = current_time

        # Only process where both are valid
        valid = ~np.isnan(self._smoothed) & ~np.isnan(self._target)
        if np.any(valid):
            # Calculate time-corrected alpha
            # Formula: 1.0 - pow(1.0 - alpha, dt * freq)

            # Determine attack vs release per element
            increasing = self._target[valid] > self._smoothed[valid]

            # Create alpha array with attack/release values
            alpha = np.where(increasing, self._attack, self._release)

            # Apply time correction (alpha is now per-second, dt converts to actual time)
            time_corrected_alpha = 1.0 - np.power(1.0 - alpha, dt)

            # Apply EMA smoothing: value += (target - value) * alpha
            error = self._target[valid] - self._smoothed[valid]
            self._smoothed[valid] += error * time_corrected_alpha

        # Apply constraints after smoothing
        self._apply_constraints()

    def _apply_constraints(self) -> None:
        """Apply value constraints (clamping for vectors, overridden for angles)."""
        if self._clamp_range is not None:
            np.clip(self._smoothed, self._clamp_range[0], self._clamp_range[1], out=self._smoothed)

    def reset(self, initial_value: float = 0.0) -> None:
        """Reset the filter to initial state."""
        self._smoothed = np.full(self._vector_size, float(initial_value))
        self._target = np.full(self._vector_size, np.nan)
        self._last_update_time = None

    @property
    def value(self) -> np.ndarray:
        """Get the current smoothed value (returns a copy)."""
        return self._smoothed.copy()

    @property
    def target(self) -> np.ndarray:
        """Get the current target value (returns a copy)."""
        return self._target.copy()

    @property
    def attack(self) -> float:
        """Get the attack smoothing factor."""
        return self._attack
    @attack.setter
    def attack(self, value: float) -> None:
        """Set the attack smoothing factor."""
        if value <= 0 or value > 1.0:
            raise ValueError("attack must be in (0.0, 1.0]")
        self._attack = float(value)

    @property
    def release(self) -> float:
        """Get the release smoothing factor."""
        return self._release
    @release.setter
    def release(self, value: float) -> None:
        """Set the release smoothing factor."""
        if value <= 0 or value > 1.0:
            raise ValueError("release must be in (0.0, 1.0]")
        self._release = float(value)

    @property
    def clamp_range(self) -> tuple[float, float] | None:
        """Get the clamp range."""
        return self._clamp_range
    @clamp_range.setter
    def clamp_range(self, value: tuple[float, float] | None) -> None:
        """Set the clamp range."""
        self._clamp_range = value

    def setParameters(self, attack: float = 0.5, release: float = 0.2) -> None:
        """Set all parameters at once.

        Args:
            attack: Smoothing factor per second for rising values (0-1)
            release: Smoothing factor per second for falling values (0-1)
        """
        self.attack = attack
        self.release = release


class AngleEMASmooth(EMASmooth):
    """EMA smoothing for angular/circular data with proper wrapping.

    Smoothing is applied to the shortest angular path between current and target values.
    """

    def __init__(self, vector_size: int, attack: float = 0.5, release: float = 0.2,
                 clamp_range: tuple[float, float] | None = None) -> None:
        """Initialize the angle EMA smoother.

        Args:
            vector_size: Number of angles to smooth
            attack: Smoothing factor per second for rising values (0-1). Higher = more responsive.
            release: Smoothing factor per second for falling values (0-1). Higher = more responsive.
            clamp_range: Ignored for angles (wrapping is always applied)
        """
        super().__init__(vector_size, attack, release, clamp_range=None)

    def update(self, values: np.ndarray, current_time: float | None = None) -> None:
        """Apply EMA smoothing to angles, calculating shortest angular path."""
        if values.shape[0] != self._vector_size:
            raise ValueError(f"Expected array of size {self._vector_size}, got {values.shape[0]}")

        target = values.copy()

        # Only calculate shortest path where both current and new values are valid
        valid = np.isfinite(self._smoothed) & np.isfinite(values)

        if np.any(valid):
            # Calculate shortest angular path for valid components only
            delta = values[valid] - self._smoothed[valid]
            delta = np.arctan2(np.sin(delta), np.cos(delta))
            target[valid] = self._smoothed[valid] + delta

        # For components where _smoothed is NaN but values is valid,
        # target already contains the raw values (direct initialization)
        super().update(target, current_time)

    def _apply_constraints(self) -> None:
        """Apply angular wrapping to [-π, π]."""
        self._smoothed = np.arctan2(np.sin(self._smoothed), np.cos(self._smoothed))


class PointEMASmooth(EMASmooth):
    """EMA smoothing for 2D points with (x, y) coordinates."""

    def __init__(self, vector_size: int, attack: float = 0.5, release: float = 0.2,
                 clamp_range: tuple[float, float] | None = None) -> None:
        """Initialize the point EMA smoother.

        Args:
            vector_size: Number of points to smooth
            attack: Smoothing factor per second for rising values (0-1). Higher = more responsive.
            release: Smoothing factor per second for falling values (0-1). Higher = more responsive.
            clamp_range: Optional (min, max) tuple to clamp point coordinates
        """
        super().__init__(vector_size * 2, attack, release, clamp_range)
        self._num_points: int = vector_size

    def update(self, points: np.ndarray, current_time: float | None = None) -> None:
        """Apply EMA smoothing to points."""
        if points.shape != (self._num_points, 2):
            raise ValueError(f"Expected shape ({self._num_points}, 2), got {points.shape}")
        super().update(points.copy().flatten(), current_time)

    @property
    def value(self) -> np.ndarray:
        """Get the current smoothed points (returns a copy)."""
        return self._smoothed.reshape(self._num_points, 2).copy()


ArrayEMASmooth = Union[EMASmooth, AngleEMASmooth, PointEMASmooth]