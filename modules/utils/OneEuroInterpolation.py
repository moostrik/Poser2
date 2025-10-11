"""
SmoothInterpolation Module

Provides a collection of interpolation classes based on the 1€ Filter algorithm
for smoothing various types of motion data. It provides interpolatedvalues between
samples using Hermite interpolation. It also handles NaN values by substituting the
last valid value.

Classes:
    OneEuroInterpolator: Basic interpolator for numeric values
    AngleEuroInterpolator: Specialized interpolator for angular data in [-π,π] range
    --ArrayEuroInterpolator: Interpolator for multi-dimensional numpy arrays--
"""

import numpy as np
from dataclasses import dataclass, field
from OneEuroFilter import OneEuroFilter
from collections import deque
from time import time
from typing import List, Callable, Any


@dataclass
class OneEuroSettings:
    """Configuration parameters for the 1€ Filter."""
    frequency: float = 30.0     # Sampling frequency (Hz)
    min_cutoff: float = 1.0     # Minimum cutoff frequency
    beta: float = 0.0           # Speed coefficient
    d_cutoff: float = 1.0       # Cutoff frequency for derivative

    def __post_init__(self) -> None:
        # Initialize observers after dataclass fields are set
        self._observers: List[Callable[[], None]] = []

    def __setattr__(self, name: str, value: Any) -> None:
        """Notify observers when attributes change"""
        super().__setattr__(name, value)
        # Only notify after initialization is complete
        if name != '_observers' and hasattr(self, '_observers'):
            self._notify()

    def add_observer(self, callback: Callable[[], None]) -> None:
        """Add a callback function to be called when settings change"""
        self._observers.append(callback)

    def remove_observer(self, callback: Callable[[], None]) -> None:
        """Remove a callback function"""
        if callback in self._observers:
            self._observers.remove(callback)

    def _notify(self) -> None:
        """Notify all observers of a change"""
        for callback in self._observers:
            callback()

class OneEuroInterpolator:
    """Basic numeric value interpolator using 1€ Filter."""

    def __init__(self, settings: OneEuroSettings) -> None:
        self._settings: OneEuroSettings = settings  # Store settings object
        self._filter: OneEuroFilter = OneEuroFilter(settings.frequency, settings.min_cutoff, settings.beta, settings.d_cutoff)
        self._interval: float = 1.0 / settings.frequency               # Expected time between samples
        self._buffer: deque[float] = deque(maxlen=4)     # Store recent filtered values
        self._last_time: float = time()                  # Timestamp of last sample
        self._last_real: float | None = None             # Last non-NaN value

        self._smooth_value: float | None = None
        self._smooth_delta: float | None = None

        settings.add_observer(self._update_filter_from_settings)

    @property
    def smooth_value(self) -> float | None:
        return self._smooth_value

    @property
    def smooth_delta(self) -> float | None:
        return self._smooth_delta

    def _update_filter_from_settings(self) -> None:
        """Update filter parameters from current settings"""
        self._filter.setMinCutoff(self._settings.min_cutoff)
        self._filter.setBeta(self._settings.beta)
        self._filter.setDerivateCutoff(self._settings.d_cutoff)

    def add_sample(self, value: float) -> None:
        if np.isnan(value):
            if self._last_real is None:
                return
            value = self._last_real
        else:
            self._last_real = value

        smoothed: float = self._filter(value)
        self._buffer.append(smoothed)
        self._last_time = time()

    def update(self) -> None:
        """
        Get an interpolated value based on current time and previously added samples.

        This method calculates an interpolated value by determining how much time has elapsed
        since the last sample was added, selecting an appropriate interpolation algorithm:
        - If more time than one interval has passed: returns the last buffer value
        - If 4 samples are available: uses cubic Hermite interpolation for smooth curves
        - If 2+ samples are available: uses linear interpolation
        - If only 1 sample: returns that sample directly

        The interpolation provides smooth transitions between sampled points while preserving
        the filtering effect of the 1€ Filter.
        """

        if not self._buffer:
            return None

        reference_time: float = time()
        alpha: float = (reference_time - self._last_time) / self._interval
        if alpha > 1.0:
            value: float = self._buffer[-1]
        elif len(self._buffer) > 3:
            value: float = OneEuroInterpolator.cubic_hermite_interpolate(self._buffer, alpha)
        elif len(self._buffer) > 1:
            value: float = OneEuroInterpolator.linear_interpolate(self._buffer, alpha)
        else:
            value = self._buffer[0]

        if self._smooth_value is None:
            self._smooth_value = value

        self._smooth_delta = value - self._smooth_value
        self._smooth_value = value

    def reset(self) -> None:
        self._buffer.clear()
        self._last_real = None
        self._last_time = time()
        self._filter.reset()

    @staticmethod
    def linear_interpolate(buffer: deque[float], fraction: float) -> float:
        """Linear interpolation between start and end by fraction (0.0-1.0)"""
        start: float = buffer[-2]
        end: float = buffer[-1]
        return start + fraction * (end - start)

    @staticmethod
    def cubic_hermite_interpolate(buffer: deque[float], alpha: float) -> float:
        """Cubic Hermite interpolation with Catmull-Rom style tangents.

        Interpolates between p2 and p3 using p0 and p1 to estimate tangents.
        This creates smooth C1 continuous curves through the sample points.

        Args:
            buffer: Deque containing at least 4 points [p0, p1, p2, p3]
            alpha: Fraction between p2 and p3 (0.0-1.0)

        Returns:
            Interpolated value
        """
        if len(buffer) < 4:
            raise ValueError("At least 3 points are required for cubic Hermite interpolation")

        p0, p1, p2, p3 = list(buffer)[-4:]

        # Calculate velocities (approximations based on equal time intervals)
        v1: float = p2 - p1
        v0: float = p1 - p0

        # Calculate Hermite basis functions
        h00: float = 2*alpha**3 - 3*alpha**2 + 1    # position from p2
        h10: float = alpha**3 - 2*alpha**2 + alpha      # velocity from p2
        h01: float = -2*alpha**3 + 3*alpha**2       # position from p3
        h11: float = alpha**3 - alpha**2            # velocity from p3

        # Use p2, p3 and their estimated velocities
        return h00*p2 + h10*v1 + h01*p3 + h11*v1

class AngleEuroInterpolator:
    """Interpolator for angular data in [-π,π] range."""

    def __init__(self, settings: OneEuroSettings) -> None:
        self._sin_interp = OneEuroInterpolator(settings)
        self._cos_interp = OneEuroInterpolator(settings)
        self._smooth_value: float | None = None
        self._smooth_delta: float | None = None

    @property
    def smooth_value(self) -> float | None:
        return self._smooth_value

    @property
    def smooth_delta(self) -> float | None:
        return self._smooth_delta

    def add_sample(self, angle: float) -> None:
        """Add an angular sample in [-π,π] range"""
        sin_val: float = np.sin(angle)
        cos_val: float = np.cos(angle)

        self._sin_interp.add_sample(sin_val)
        self._cos_interp.add_sample(cos_val)

    def update(self) -> None:
        """Get interpolated angle in [-π,π] range"""
        self._sin_interp.update()
        self._cos_interp.update()

        sin_val: float | None = self._sin_interp._smooth_value
        cos_val: float | None = self._cos_interp._smooth_value

        if sin_val is None or cos_val is None:
            return None

        value: float = float(np.arctan2(sin_val, cos_val))
        if self._smooth_value is None:
            self._smooth_value = value
        self._smooth_delta = float(np.mod(value - self._smooth_value + np.pi, 2 * np.pi) - np.pi)
        self._smooth_value = value

    def reset(self) -> None:
        self._sin_interp.reset()
        self._cos_interp.reset()

# class ArrayEuroInterpolator:
#     """Interpolator for multi-dimensional arrays."""

#     def __init__(self, freq:float, settings: OneEuroSettings, shape: tuple[int, ...], angular: bool=False, normalize: bool=False) -> None:
#         """
#         Args:
#             freq: Sampling frequency (Hz)
#             settings: Filter parameters
#             shape: Shape of arrays to filter
#             angular: Whether to use angle interpolation
#             normalize: Whether to normalize values to [0,1]
#         """

#         self.shape: tuple[int, ...] = shape
#         self.size: int = int(np.prod(shape))
#         self.normalize: bool = normalize

#         self.interpolators: list[OneEuroInterpolator | AngleEuroInterpolator]

#         if angular:
#             self.interpolators = [AngleEuroInterpolator(settings)
#                                  for _ in range(self.size)]
#         else:
#             self.interpolators = [OneEuroInterpolator(settings)
#                                  for _ in range(self.size)]

#     def add_sample(self, array: np.ndarray) -> None:
#         flat_array: np.ndarray = array.flatten()

#         if len(flat_array) != self.size:
#             raise ValueError(f"Expected shape: {self.shape}, received array of shape {array.shape}")

#         if self.normalize:
#             flat_array = np.clip(flat_array, 0.0, 1.0)

#         for i, value in enumerate(flat_array):
#             self.interpolators[i].add_sample(value)

#     def get(self) -> np.ndarray:
#         result: np.ndarray = np.zeros(self.size)
#         for i, interp in enumerate(self.interpolators):
#             val: float | None = interp.get()
#             result[i] = val if val is not None else 0.0

#         if self.normalize:
#             result = np.clip(result, 0.0, 1.0)

#         return result.reshape(self.shape)

#     def reset(self) -> None:
#         for interp in self.interpolators:
#             interp.reset()
