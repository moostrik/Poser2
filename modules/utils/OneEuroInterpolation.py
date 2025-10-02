
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
from dataclasses import dataclass
from OneEuroFilter import OneEuroFilter
from collections import deque
from time import time

@dataclass
class OneEuroSettings:
    """Configuration parameters for the 1€ Filter."""
    min_cutoff: float = 1.0     # Minimum cutoff frequency
    beta: float = 0.0           # Speed coefficient
    d_cutoff: float = 1.0       # Cutoff frequency for derivative

class OneEuroInterpolator:
    """Basic numeric value interpolator using 1€ Filter."""

    def __init__(self, freq: float, settings: OneEuroSettings) -> None:
        self.filter: OneEuroFilter = OneEuroFilter(freq, settings.min_cutoff, settings.beta, settings.d_cutoff)
        self.interval: float = 1.0 / freq               # Expected time between samples
        self.buffer: deque[float] = deque(maxlen=4)     # Store recent filtered values
        self.last_time: float = time()                  # Timestamp of last sample
        self.last_real: float | None = None             # Last non-NaN value

    def update_settings(self, settings: OneEuroSettings) -> None:
        self.filter.setMinCutoff(settings.min_cutoff)
        self.filter.setBeta(settings.beta)
        self.filter.setDerivateCutoff(settings.d_cutoff)

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

    def get(self) -> float | None:
        if not self.buffer:
            return None

        if len(self.buffer) == 1:
            return self.buffer[0]

        time_delta: float = time() - self.last_time
        alpha: float = min(time_delta / self.interval, 1.5)  # Limit extrapolation factor


        if len(self.buffer) >= 3:
            # Get last 3 points for interpolation
            if len(self.buffer) >= 4:
                p0, p1, p2, p3 = list(self.buffer)[-4:]
            else:
                # If only 3 points, duplicate first point
                p0 = list(self.buffer)[-3]
                p1, p2, p3 = list(self.buffer)[-3:]

            # Calculate velocities (approximations based on equal time intervals)
            v1: float = p2 - p1
            v0: float = p1 - p0

            # Calculate acceleration
            accel: float = v1 - v0

            # Catmull-Rom style interpolation between p2 and p3
            if alpha < 1.0:
                # Interpolate between p2 and p3 with velocity influence
                t: float = alpha
                # Calculate Hermite basis functions
                h00: float = 2*t**3 - 3*t**2 + 1    # position from p2
                h10: float = t**3 - 2*t**2 + t      # velocity from p2
                h01: float = -2*t**3 + 3*t**2       # position from p3
                h11: float = t**3 - t**2            # velocity from p3

                # Use p2, p3 and their estimated velocities
                return h00*p2 + h10*v1 + h01*p3 + h11*v1
            else:
                # Extrapolation case - use velocity and acceleration from last points
                # Limit extrapolation to reduce instability
                t = min(alpha - 1.0, 0.5)
                return p3 + v1*t + 0.5*accel*t*t
        else:
            # Only have two points - enhanced extrapolation
            v0, v1 = self.buffer[-2], self.buffer[-1]
            velocity: float = v1 - v0

            if alpha < 1.0:
                # Simple linear interpolation between the two points
                return v0 + alpha * velocity
            else:
                # Extrapolation with velocity decay
                t = alpha - 1.0
                return v1 + velocity * t
                # Apply velocity decay factor (0.8) to simulate natural deceleration
                # decay_factor = 1.0 / (1.0 + 0.5 * t)
                # return v1 + velocity * t * decay_factor

class NormalizedEuroInterpolator:
    """Interpolator for values in [0,1] range."""

    def __init__(self, freq: float, settings: OneEuroSettings) -> None:
        self.interp = OneEuroInterpolator(freq, settings)

    def update_settings(self, settings: OneEuroSettings) -> None:
        self.interp.update_settings(settings)

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

class AngleEuroInterpolator:
    """Interpolator for angular data in [-π,π] range."""

    def __init__(self, freq: float, settings: OneEuroSettings) -> None:
        self.sin_interp = OneEuroInterpolator(freq, settings)
        self.cos_interp = OneEuroInterpolator(freq, settings)

    def update_settings(self, settings: OneEuroSettings) -> None:
        self.sin_interp.update_settings(settings)
        self.cos_interp.update_settings(settings)

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
            self.interpolators = [AngleEuroInterpolator(freq, settings)
                                 for _ in range(self.size)]
        else:
            self.interpolators = [OneEuroInterpolator(freq, settings)
                                 for _ in range(self.size)]

    def update_settings(self, settings: OneEuroSettings) -> None:
        for interp in self.interpolators:
            interp.update_settings(settings)

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
