"""
SmoothInterpolation Module

Provides a collection of interpolation classes based on the 1€ Filter algorithm
for smoothing various types of motion data. It provides interpolatedvalues between
samples using Hermite interpolation. It also handles NaN values by substituting the
last valid value.

Classes:
    OneEuroInterpolator: Basic interpolator for numeric values
    NormalizedEuroInterpolator: Interpolator for values in [0,1] range
    AngleEuroInterpolator: Specialized interpolator for angular data in [-π,π] range
    ArrayEuroInterpolator: Interpolator for multi-dimensional numpy arrays
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
        self.settings: OneEuroSettings = settings  # Store settings object
        self.filter: OneEuroFilter = OneEuroFilter(settings.frequency, settings.min_cutoff, settings.beta, settings.d_cutoff)
        self.interval: float = 1.0 / settings.frequency               # Expected time between samples
        self.buffer: deque[float] = deque(maxlen=5)     # Store recent filtered values
        self.last_time: float = time()                  # Timestamp of last sample
        self.last_real: float | None = None             # Last non-NaN value

        settings.add_observer(self._update_filter_from_settings)

    def _update_filter_from_settings(self) -> None:
        """Update filter parameters from current settings"""
        self.filter.setMinCutoff(self.settings.min_cutoff)
        self.filter.setBeta(self.settings.beta)
        self.filter.setDerivateCutoff(self.settings.d_cutoff)

    def add_sample(self, value: float) -> None:
        if np.isnan(value):
            if self.last_real is None:
                return
            value = self.last_real
        else:
            self.last_real = value

        smoothed: float = self.filter(value)
        self.buffer.append(smoothed)
        self.last_time = time()

    def get(self, timestamp: float | None = None) -> float | None:
        """
        Get interpolated value at the specified timestamp or current time.

        Args:
            timestamp:  Optional timestamp to calculate value at.
                        works within the range of last two samples.
                        If None, uses current time (original behavior).
        """
        if not self.buffer:
            return None

        if len(self.buffer) == 1:
            return self.buffer[0]

        # Use provided timestamp or current time
        reference_time: float = timestamp if timestamp is not None else time()
        if reference_time > self.last_time:
            alpha: float = (reference_time - self.last_time) / self.interval
            if alpha > 1.0:
                return self.buffer[-1]

            if len(self.buffer) == 2:
                return OneEuroInterpolator.linear_interpolate(self.buffer[-2], self.buffer[-1], alpha)
            if len(self.buffer) == 3:
                # Simple quadratic interpolation using last 3 points
                p1, p2, p3 = list(self.buffer)[-3:]
                p0 = p1  # Duplicate first point for lack of a fourth
                return OneEuroInterpolator.cubic_hermite_interpolate(p0, p1, p2, p3, alpha)
            if len(self.buffer) > 3:
                # Cubic Hermite interpolation using last 4 points
                p0, p1, p2, p3 = list(self.buffer)[-4:]
                return OneEuroInterpolator.cubic_hermite_interpolate(p0, p1, p2, p3, alpha)

        else: # reference_time <= self.last_time
            alpha: float = (reference_time - (self.last_time - self.interval)) / self.interval
            if len(self.buffer) == 2 or alpha <= 0.0:
                return self.buffer[-2] # same as [0]
            if len(self.buffer) == 3:
                return OneEuroInterpolator.linear_interpolate(self.buffer[-3], self.buffer[-2], alpha)
            if len(self.buffer) == 4:
                p0, p1, p2 = list(self.buffer)[-4:-1]
                p3 = p2  # Duplicate last point for lack of a fourth
                return OneEuroInterpolator.cubic_hermite_interpolate(p0, p1, p2, p3, alpha)
            if len(self.buffer) > 4:
                p0, p1, p2, p3 = list(self.buffer)[-5:-1]
                return OneEuroInterpolator.cubic_hermite_interpolate(p0, p1, p2, p3, alpha)


    def get_delta(self, start_time: float, end_time: float) -> float:
        """Get change in value since last sample at the specified timestamp."""
        if not self.buffer or len(self.buffer) < 2:
            return 0.0

        start_value: float | None = self.get(start_time)
        end_value: float | None = self.get(end_time)
        if start_value is None or end_value is None:
            return 0.0
        return end_value - start_value

    def reset(self) -> None:
        self.buffer.clear()
        self.last_real = None
        self.last_time = time()
        self.filter.reset()

    @staticmethod
    def linear_interpolate(start: float, end: float, fraction: float) -> float:
        """Linear interpolation between start and end by fraction (0.0-1.0)"""
        return start + fraction * (end - start)

    @staticmethod
    def cubic_hermite_interpolate(p0: float, p1: float, p2: float, p3: float, t: float) -> float:
        """Cubic Hermite interpolation with Catmull-Rom style tangents.

        Interpolates between p2 and p3 using p0 and p1 to estimate tangents.
        This creates smooth C1 continuous curves through the sample points.

        Args:
            p0: First control point (earliest)
            p1: Second control point
            p2: Third control point (interpolation starts here)
            p3: Fourth control point (interpolation ends here)
            t: Interpolation parameter [0.0-1.0]

        Returns:
            Interpolated value
        """
        # Calculate velocities (approximations based on equal time intervals)
        v1: float = p2 - p1
        v0: float = p1 - p0

        # Calculate Hermite basis functions
        h00: float = 2*t**3 - 3*t**2 + 1    # position from p2
        h10: float = t**3 - 2*t**2 + t      # velocity from p2
        h01: float = -2*t**3 + 3*t**2       # position from p3
        h11: float = t**3 - t**2            # velocity from p3

        # Use p2, p3 and their estimated velocities
        return h00*p2 + h10*v1 + h01*p3 + h11*v1

class NormalizedEuroInterpolator:
    """Interpolator for values in [0,1] range."""

    def __init__(self, settings: OneEuroSettings) -> None:
        self.interp = OneEuroInterpolator(settings)

    def add_sample(self, value: float) -> None:
        """Add an sample in [0,1] range"""
        clamped_value: float = max(0.0, min(1.0, value))
        self.interp.add_sample(clamped_value)

    def get(self) -> float | None:
        """Get interpolated value in [0,1] range"""
        val: float | None = self.interp.get()
        if val is None:
            return None
        return max(0.0, min(1.0, val))

    def reset(self) -> None:
        self.interp.reset()

class AngleEuroInterpolator:
    """Interpolator for angular data in [-π,π] range."""

    def __init__(self, settings: OneEuroSettings) -> None:
        self.sin_interp = OneEuroInterpolator(settings)
        self.cos_interp = OneEuroInterpolator(settings)

    def add_sample(self, angle: float) -> None:
        """Add an angular sample in [-π,π] range"""
        sin_val: float = np.sin(angle)
        cos_val: float = np.cos(angle)

        self.sin_interp.add_sample(sin_val)
        self.cos_interp.add_sample(cos_val)

    def get(self) -> float | None:
        """Get interpolated angle in [-π,π] range"""
        sin_val: float | None = self.sin_interp.get()
        cos_val: float | None = self.cos_interp.get()

        if sin_val is None or cos_val is None:
            return None

        return np.arctan2(sin_val, cos_val)

    def reset(self) -> None:
        self.sin_interp.reset()
        self.cos_interp.reset()

class ArrayEuroInterpolator:
    """Interpolator for multi-dimensional arrays."""

    def __init__(self, freq:float, settings: OneEuroSettings, shape: tuple[int, ...], angular: bool=False, normalize: bool=False) -> None:
        """
        Args:
            freq: Sampling frequency (Hz)
            settings: Filter parameters
            shape: Shape of arrays to filter
            angular: Whether to use angle interpolation
            normalize: Whether to normalize values to [0,1]
        """

        self.shape: tuple[int, ...] = shape
        self.size: int = int(np.prod(shape))
        self.normalize: bool = normalize

        self.interpolators: list[OneEuroInterpolator | AngleEuroInterpolator]

        if angular:
            self.interpolators = [AngleEuroInterpolator(settings)
                                 for _ in range(self.size)]
        else:
            self.interpolators = [OneEuroInterpolator(settings)
                                 for _ in range(self.size)]

    def add_sample(self, array: np.ndarray) -> None:
        flat_array: np.ndarray = array.flatten()

        if len(flat_array) != self.size:
            raise ValueError(f"Expected shape: {self.shape}, received array of shape {array.shape}")

        if self.normalize:
            flat_array = np.clip(flat_array, 0.0, 1.0)

        for i, value in enumerate(flat_array):
            self.interpolators[i].add_sample(value)

    def get(self) -> np.ndarray:
        result: np.ndarray = np.zeros(self.size)
        for i, interp in enumerate(self.interpolators):
            val: float | None = interp.get()
            result[i] = val if val is not None else 0.0

        if self.normalize:
            result = np.clip(result, 0.0, 1.0)

        return result.reshape(self.shape)

    def reset(self) -> None:
        for interp in self.interpolators:
            interp.reset()
