"""
OneEuroInterpolation Module

Smooths noisy input signals and generates interpolated values between samples.

Combines two techniques:
- 1€ Filter: Adaptive low-pass filter that reduces noise while tracking fast movements
- Hermite interpolation: Generates smooth values between filtered samples

Purpose: Convert low-framerate noisy input (e.g., 30 FPS pose detection) into
smooth high-framerate output (e.g., 60 FPS animation) with minimal lag.

How it works:
1. add_sample() filters input with 1€ Filter and buffers result (input frame rate)
2. update() interpolates between buffered samples based on elapsed time (output frame rate)
3. Output is smooth, low-noise signal queryable at any framerate

Classes:
    OneEuroSettings: Filter configuration parameters
    SmoothedInterpolator: Scalar value smoother/interpolator
    SmoothedAngleInterpolator: Angular value smoother (handles ±π wraparound)

Usage Example:
    # IMPORTANT: Set frequency to INPUT rate, not output rate!
    settings = OneEuroSettings(frequency=30.0, min_cutoff=1.0, beta=0.0)
    interp = SmoothedInterpolator(settings)

    # Feed at input rate (30 Hz)
    def on_pose_update(value):
        interp.add_sample(value)

    # Query at output rate (60 Hz)
    def render_frame():
        interp.update()
        draw(interp.smooth_value)

Notes:
    - frequency parameter must match input rate, not output rate
    - Not thread-safe
    - NaN inputs use last valid value; output is NaN until first valid sample
    - Pass current_time to update() when using multiple interpolators for efficiency
"""

import math
from dataclasses import dataclass
from OneEuroFilter import OneEuroFilter
from collections import deque
from time import time
from typing import Callable

from modules.utils.HotReloadMethods import HotReloadMethods


@dataclass
class OneEuroSettings:
    """Configuration parameters for the 1€ Filter.

    Attributes:
        frequency: INPUT data frequency in Hz (default: 30.0)
                  This is the frequency at which add_sample() is called.
                  Example: For 30 FPS pose detection, use frequency=30.0
                          Then call update() at any desired rate (e.g., 60 FPS)
        min_cutoff: Minimum cutoff frequency for position smoothing (default: 1.0)
        beta: Speed coefficient - higher values reduce lag but increase jitter (default: 0.0)
        d_cutoff: Cutoff frequency for derivative smoothing (default: 1.0)
    """
    frequency: float = 30.0
    min_cutoff: float = 1.0
    beta: float = 0.0
    d_cutoff: float = 1.0

    def __post_init__(self) -> None:
        """Initialize observer list and validate parameters."""
        # Validate parameters
        if self.frequency <= 0:
            raise ValueError(f"frequency must be > 0, got {self.frequency}")
        if self.min_cutoff < 0:
            raise ValueError(f"min_cutoff must be non-negative, got {self.min_cutoff}")
        if self.beta < 0:
            raise ValueError(f"beta must be non-negative, got {self.beta}")
        if self.d_cutoff < 0:
            raise ValueError(f"d_cutoff must be non-negative, got {self.d_cutoff}")

        self._observers: list[Callable[[], None]] = []

    def __setattr__(self, name: str, value: object) -> None:
        """Notify observers when settings change."""
        super().__setattr__(name, value)
        # Only notify after full initialization and for non-internal attributes
        if name != '_observers' and hasattr(self, '_observers') and len(self._observers) > 0:
            self._notify()

    def add_observer(self, callback: Callable[[], None]) -> None:
        """Add observer callback to be notified of setting changes.

        Args:
            callback: Function to call when settings change
        """
        self._observers.append(callback)

    def remove_observer(self, callback: Callable[[], None]) -> None:
        """Remove observer callback.

        Args:
            callback: Previously registered callback function
        """
        if callback in self._observers:
            self._observers.remove(callback)

    def _notify(self) -> None:
        """Notify all observers of setting changes."""
        for callback in self._observers:
            callback()


class SmoothedInterpolator:
    """Basic numeric value interpolator using 1€ Filter with Hermite interpolation.

    Smooths incoming samples using a 1€ Filter, then interpolates between the
    two most recent samples using cubic Hermite splines. Also tracks velocity
    and acceleration of the smoothed signal.

    The interpolation provides smooth animation between samples with minimal
    latency (approximately 1 frame).

    Attributes:
        smooth_value: Current interpolated value
        smooth_velocity: Current velocity (change per update, not per second)
        smooth_acceleration: Current acceleration (change in velocity per update)
        settings: Configuration settings (modifiable)
    """

    def __init__(self, settings: OneEuroSettings) -> None:
        """Initialize interpolator with settings.

        Args:
            settings: OneEuroSettings configuration object
        """
        self._settings: OneEuroSettings = settings
        self._filter: OneEuroFilter = OneEuroFilter(
            settings.frequency,
            settings.min_cutoff,
            settings.beta,
            settings.d_cutoff
        )
        self._interval: float = 1.0 / settings.frequency
        self._buffer: deque[float] = deque(maxlen=4)
        self._last_time: float = time()
        self._last_valid_input_value: float | None = None

        # Initialize state as NaN (no data yet)
        self._smooth_value: float = math.nan
        self._smooth_velocity: float = math.nan
        self._smooth_acceleration: float = math.nan

        # Register for setting changes
        settings.add_observer(self._update_filter_from_settings)

        self.hotreload = HotReloadMethods(self.__class__, True, True)

    def __del__(self) -> None:
        """Cleanup observer registration on deletion."""
        try:
            self._settings.remove_observer(self._update_filter_from_settings)
        except (AttributeError, ValueError):
            pass  # Already cleaned up or settings deleted

    def __repr__(self) -> str:
        """Debug representation."""
        value_str: str = f"{self._smooth_value:.3f}" if not math.isnan(self._smooth_value) else "NaN"
        return f"OneEuroInterpolator(value={value_str}, samples={len(self._buffer)}/4)"

    @property
    def smooth_value(self) -> float:
        """Current interpolated value. NaN if no samples received yet."""
        return self._smooth_value

    @property
    def smooth_velocity(self) -> float:
        """Current velocity as difference per update (not per second). NaN if no samples yet."""
        return self._smooth_velocity

    @property
    def smooth_acceleration(self) -> float:
        """Current acceleration as change in velocity per update. NaN if no samples yet."""
        return self._smooth_acceleration

    @property
    def settings(self) -> OneEuroSettings:
        """Configuration settings. Modifications trigger filter updates."""
        return self._settings

    def _update_filter_from_settings(self) -> None:
        """Update filter parameters when settings change."""
        self._filter.setMinCutoff(self._settings.min_cutoff)
        self._filter.setBeta(self._settings.beta)
        self._filter.setDerivateCutoff(self._settings.d_cutoff)

    def add_sample(self, value: float, timestamp: float | None = None) -> None:
        """Add a sample value to the interpolator.

        NaN values are substituted with the last valid value to prevent NaN
        propagation through the filter. If no valid value exists yet, NaN
        samples are skipped.

        Args:
            value: Sample value to add (may be NaN)
            timestamp: Optional timestamp in seconds. If None, uses time().
                      Providing timestamp improves performance when adding
                      multiple samples at the same time.
        """
        if math.isnan(value):
            if self._last_valid_input_value is None:
                return  # Skip if no valid history
            value = self._last_valid_input_value
        else:
            self._last_valid_input_value = value

        # Apply 1€ filter smoothing
        smoothed: float = self._filter(value)
        self._buffer.append(smoothed)
        self._last_time = timestamp if timestamp is not None else time()

    def update(self, current_time: float | None = None) -> None:
        self._update(current_time)

    def _update(self, current_time: float | None = None) -> None:
        """Calculate interpolated value and derivatives for current time."""
        if not self._buffer:
            return

        if current_time is None:
            current_time = time()

        alpha: float = (current_time - self._last_time) / self._interval

        # Select interpolation method
        if alpha <= 0.0 or alpha >= 1.0:
            value: float = self._buffer[-1]
            velocity: float = (self._buffer[-1] - self._buffer[-2]) / self._interval if len(self._buffer) >= 2 else 0.0
        # elif len(self._buffer) >= 4:
        #     # Use Hermite interpolation for smooth position AND velocity
        #     value: float = self._hermite_interpolate_latest(alpha)
        #     velocity: float = self._hermite_velocity_latest(alpha)
        elif len(self._buffer) >= 3:
            # Fallback to linear if not enough samples
            p1: float = self._buffer[-3]
            p2: float = self._buffer[-2]
            p3: float = self._buffer[-1]
            value: float = p2 + alpha * (p3 - p2)

            v2: float = (p2 - p1)  # Velocity at p2
            v3: float = (p3 - p2)  # Velocity at p3
            velocity: float = p1
        else:
            value = self._buffer[0]
            velocity = 0.0

        # Initialize on first valid value
        if math.isnan(self._smooth_value):
            self._smooth_value = value
            self._smooth_velocity = velocity
            return

        # Update state
        self._smooth_value = value
        self._smooth_velocity = 0.1

    def reset(self, current_time: float | None = None) -> None:
        """Reset interpolator to initial state.

        Clears all buffered samples and resets state to NaN. Useful when
        starting a new tracking sequence.

        Args:
            current_time: Optional timestamp in seconds. If None, uses time().
        """
        self._buffer.clear()
        self._last_valid_input_value = None
        self._last_time = current_time if current_time is not None else time()
        self._filter.reset()

        # Reset to NaN (no data)
        self._smooth_value = math.nan
        self._smooth_velocity = math.nan
        self._smooth_acceleration = math.nan

    def _linear_interpolate_latest(self, alpha: float) -> float:
        """Linear interpolation between the two most recent samples.

        Args:
            alpha: Interpolation factor [0, 1] where 0=second-to-last, 1=last

        Returns:
            Interpolated value between the two most recent samples
        """
        p2: float = self._buffer[-2]  # Second-to-last
        p3: float = self._buffer[-1]  # Latest
        return p2 + alpha * (p3 - p2)

    def _hermite_interpolate_latest(self, alpha: float) -> float:
        """Cubic Hermite interpolation between the two most recent samples.

        Uses a 4-point stencil to create smooth C1-continuous interpolation
        between the two most recent samples (p2 and p3), using the two
        older samples (p0 and p1) to estimate tangents.

        Tangent at p2: Centered Catmull-Rom difference (p3 - p1) / 2
        Tangent at p3: Centered Catmull-Rom difference (p3 - p2), assuming continuation

        Args:
            alpha: Interpolation factor [0, 1] where 0=p2, 1=p3

        Returns:
            Smoothly interpolated value between p2 and p3
        """
        p0: float = self._buffer[-4]
        p1: float = self._buffer[-3]
        p2: float = self._buffer[-2]
        p3: float = self._buffer[-1]

        # Tangents using Catmull-Rom scheme
        m2: float = (p3 - p1) * 0.5  # Centered difference at p2
        m3: float = (p3 - p2) * 0.5  # Forward-looking: assume same velocity continues

        # Alternative: Use centered difference assuming p3 velocity continues
        m3: float = (p3 - p2) * 0.5 + (p3 - p2) * 0.5  # = (p3 - p2)
        # Or match the velocity from p2->p3:
        # m3: float = m2  # Copy tangent from p2 for maximum smoothness

        # Hermite basis functions
        h00: float = 2*alpha**3 - 3*alpha**2 + 1
        h10: float = alpha**3 - 2*alpha**2 + alpha
        h01: float = -2*alpha**3 + 3*alpha**2
        h11: float = alpha**3 - alpha**2

        return h00*p2 + h10*m2 + h01*p3 + h11*m3

    def _hermite_velocity_latest(self, alpha: float) -> float:
        """Velocity (first derivative) of Hermite interpolation.

        Args:
            alpha: Interpolation factor [0, 1]

        Returns:
            Velocity at interpolation point (units per interval)
        """
        p0: float = self._buffer[-4]
        p1: float = self._buffer[-3]
        p2: float = self._buffer[-2]
        p3: float = self._buffer[-1]

        # Same tangents as position interpolation
        m2: float = (p3 - p1) * 0.5
        m3: float = (p3 - p2) * 0.5  # Must match the interpolation tangent
        m3: float = (p3 - p2) * 0.5 + (p3 - p2) * 0.5  # = (p3 - p2)
        # m3: float = m2

        # Derivatives of Hermite basis functions
        dh00: float = 6*alpha**2 - 6*alpha
        dh10: float = 3*alpha**2 - 4*alpha + 1
        dh01: float = -6*alpha**2 + 6*alpha
        dh11: float = 3*alpha**2 - 2*alpha

        return (dh00*p2 + dh10*m2 + dh01*p3 + dh11*m3) / self._interval


class SmoothedAngleInterpolator:
    """Interpolator for angular data in [-π,π] range.

    Handles angular data by decomposing into sin/cos components, smoothing
    each component separately using 1€ Filters, then reconstructing the angle.
    This avoids discontinuities at the ±π boundary.

    Also tracks angular velocity and acceleration with proper wrapping.

    Attributes:
        smooth_value: Current interpolated angle in [-π,π]
        smooth_velocity: Current angular velocity in [-π,π] per update
        smooth_acceleration: Current angular acceleration in [-π,π] per update
        settings: Configuration settings (shared with underlying interpolators)
    """

    def __init__(self, settings: OneEuroSettings) -> None:
        """Initialize angle interpolator with settings.

        Args:
            settings: OneEuroSettings configuration object (shared by both components)
        """
        # Use separate interpolators for sin and cos components
        self._sin_interp = SmoothedInterpolator(settings)
        self._cos_interp = SmoothedInterpolator(settings)

        # Initialize angular state as NaN
        self._smooth_value: float = math.nan
        self._smooth_velocity: float = math.nan
        self._smooth_acceleration: float = math.nan

        self.hotreload = HotReloadMethods(self.__class__, True, True)

    def __repr__(self) -> str:
        """Debug representation."""
        if math.isnan(self._smooth_value):
            return "AngleEuroInterpolator(angle=NaN)"
        deg = math.degrees(self._smooth_value)
        return f"AngleEuroInterpolator(angle={self._smooth_value:.3f}rad, {deg:.1f}°)"

    @property
    def smooth_value(self) -> float:
        """Current interpolated angle in [-π,π]. NaN if no samples received yet."""
        return self._smooth_value

    @property
    def smooth_velocity(self) -> float:
        """Current angular velocity in [-π,π] per update. NaN if no samples yet."""
        return self._smooth_velocity

    @property
    def smooth_acceleration(self) -> float:
        """Current angular acceleration in [-π,π] per update. NaN if no samples yet."""
        return self._smooth_acceleration

    @property
    def settings(self) -> OneEuroSettings:
        """Configuration settings (shared by sin and cos interpolators)."""
        return self._sin_interp.settings

    def add_sample(self, angle: float, timestamp: float | None = None) -> None:
        """Add an angular sample in [-π,π] range.

        The angle is decomposed into sin/cos components which are smoothed
        separately to avoid wrap-around issues at ±π.

        NaN values are substituted with the last valid angle.

        Args:
            angle: Angle in radians, range [-π,π] (may be NaN)
            timestamp: Optional timestamp in seconds. If None, uses time().
                      Pass the same timestamp when adding multiple samples
                      to improve performance and consistency.
        """
        if math.isnan(angle):
            # Let underlying interpolators handle NaN via their last valid value
            self._sin_interp.add_sample(math.nan, timestamp)
            self._cos_interp.add_sample(math.nan, timestamp)
        else:
            # Decompose angle into sin/cos components
            sin_val: float = math.sin(angle)
            cos_val: float = math.cos(angle)
            self._sin_interp.add_sample(sin_val, timestamp)
            self._cos_interp.add_sample(cos_val, timestamp)

    def update(self, current_time: float | None = None) -> None:
        """Calculate interpolated angle and derivatives for current time.

        Should be called every frame. Reconstructs the angle from smoothed
        sin/cos components and calculates wrapped angular velocity and
        acceleration.

        Args:
            current_time: Optional current timestamp in seconds. If None, uses time().
                         Pass the same time when updating multiple interpolators
                         for better performance and temporal consistency.
        """
        self._update(current_time)

    def _update(self, current_time: float | None = None) -> None:
        # Update component interpolators with same timestamp
        self._sin_interp.update(current_time)
        self._cos_interp.update(current_time)

        # Use public properties instead of private members
        sin_val: float = self._sin_interp.smooth_value
        cos_val: float = self._cos_interp.smooth_value

        # Check if components are still NaN (no samples yet)
        if math.isnan(sin_val) or math.isnan(cos_val):
            return

        # Reconstruct angle from sin/cos components
        value: float = float(math.atan2(sin_val, cos_val))

        # Initialize on first valid value
        if math.isnan(self._smooth_value):
            self._smooth_value = value
            self._smooth_velocity = 0.0
            self._smooth_acceleration = 0.0
            return

        # Calculate angular velocity with wrapping to [-π, π]
        # This ensures we take the shortest path around the circle
        velocity: float = (value - self._smooth_value + math.pi) % (2 * math.pi) - math.pi

        # Calculate angular acceleration with wrapping
        acceleration: float = (velocity - self._smooth_velocity + math.pi) % (2 * math.pi) - math.pi

        # print(f"AngleInterp Update: value={value:.3f}, vel={velocity:.3f}, acc={acceleration:.3f}")

        # Update state
        self._smooth_value = value
        self._smooth_velocity = velocity
        self._smooth_acceleration = acceleration

    def reset(self, current_time: float | None = None) -> None:
        """Reset interpolator to initial state.

        Clears all buffered samples and resets state to NaN.

        Args:
            current_time: Optional timestamp in seconds. If None, uses time().
        """
        self._sin_interp.reset(current_time)
        self._cos_interp.reset(current_time)

        self._smooth_value = math.nan
        self._smooth_velocity = math.nan
        self._smooth_acceleration = math.nan
