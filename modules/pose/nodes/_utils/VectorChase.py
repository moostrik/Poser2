"""
Feel	responsiveness	friction	Result
Snappy	0.4-0.6	        0.05-0.08	Quick response, slight overshoot
Smooth	0.15-0.25	    0.02-0.04	Slower, fluid motion
Tight	0.3-0.4	        0.03-0.05	Balanced, no lag
Floaty	0.1-0.15	    0.01-0.02	Slow, dreamy motion

Algorithm Explanation:
This interpolator uses perpetual chase dynamics - it continuously moves toward a moving target
without ever fully catching it. This creates smooth, natural motion without hard stops or starts.

The system acts as a PID controller (Proportional-Derivative):
- Proportional: velocity_target = distance / input_interval (response to position error)
- Derivative: responsiveness controls rate of velocity change
- Damping: friction provides additional velocity decay

Key insight: By recalculating velocity every frame based on current distance, the system
naturally decelerates as it approaches the target, creating adaptive easing without explicit
curves. The input_interval parameter sets the time horizon - "how fast to move to reach the
target in one input period" - which scales the response appropriately for the input frequency.

Note: These classes are NOT thread-safe by design.
"""


# Standard library imports
from time import monotonic
from typing import Union

# Third-party imports
import numpy as np
import warnings

from modules.utils.HotReloadMethods import HotReloadMethods

class VectorChase:
    """Chase interpolator for arbitrary vector data (positions, coordinates, etc.)."""

    def __init__(self, vector_size: int, input_frequency: float = 30.0, responsiveness: float = 0.2,
                 friction: float = 0.03, clamp_range: tuple[float, float] | None = None) -> None:
        """Initialize the vectorized chase interpolator with velocity steering.

        This interpolator is designed for a dual-frequency architecture:
        - set_target() is called at the input frequency (e.g., 30 FPS from pose detection)
        - update() is called at the render frequency (e.g., 60+ FPS for smooth display)

        The interpolator uses perpetual chase dynamics, where it continuously moves toward
        the target without fully catching it. This creates smooth motion by:
        1. Calculating target velocity based on distance / input_interval
        2. Steering current velocity toward target velocity (responsiveness)
        3. Applying velocity damping (friction)
        4. Recalculating each frame for automatic adaptive deceleration

        The system behaves as a PID controller (Proportional-Derivative):
        - P term: Distance to target scaled by input_interval
        - D term: Responsiveness controls velocity convergence rate
        - Damping: Friction prevents oscillation

        Args:
            vector_size: Number of values to interpolate simultaneously
            input_frequency: Expected rate of set_target() calls in Hz. This sets the time
                           horizon for velocity calculation: "reach target in one input period"
            responsiveness: How quickly velocity converges to target velocity [0.0, 1.0].
                          Higher = faster response but less smooth (default: 0.2)
            friction: Velocity damping factor [0.0, 1.0]. Higher = more damping,
                     prevents overshoot (default: 0.03)
            clamp_range: Optional (min, max) tuple to clamp interpolated values

        Raises:
            ValueError: If vector_size <= 0, input_frequency <= 0, or parameters out of range
        """
        if vector_size <= 0:
            raise ValueError(f"vector_size must be positive, got {vector_size}")
        if input_frequency <= 0.0:
            raise ValueError(f"input_frequency must be positive, got {input_frequency}")
        if not 0.0 <= responsiveness <= 1.0:
            raise ValueError(f"responsiveness must be in [0.0, 1.0], got {responsiveness}")
        if not 0.0 <= friction <= 1.0:
            raise ValueError(f"friction must be in [0.0, 1.0], got {friction}")
        if clamp_range is not None:
            if len(clamp_range) != 2 or clamp_range[0] >= clamp_range[1]:
                raise ValueError("clamp_range must be (min, max) with min < max")

        self._vector_size: int = vector_size
        self._input_interval: float = 1.0 / input_frequency
        self._responsiveness: float = responsiveness
        self._inv_friction: float = 1.0 - friction
        self._clamp_range: tuple[float, float] | None = clamp_range

        self._last_update_time: float | None = None

        self._target: np.ndarray = np.full(vector_size, np.nan)
        self._interpolated: np.ndarray = np.full(vector_size, np.nan)

        self._v_curr: np.ndarray = np.zeros(vector_size)
        self._v_target: np.ndarray = np.zeros(vector_size)

    def reset(self) -> None:
        """Reset the interpolator's internal state."""
        self._last_update_time = None

        self._target = np.full(self._vector_size, np.nan)
        self._interpolated = np.full(self._vector_size, np.nan)

        self._v_curr = np.zeros(self._vector_size)
        self._v_target = np.zeros(self._vector_size)

        # hot reloader
        self.hot_reloader = HotReloadMethods(self.__class__, True, True)

    @property
    def input_frequency(self) -> float:
        """Get the input frequency (Hz) used for velocity time horizon calculation."""
        return 1.0 / self._input_interval

    @input_frequency.setter
    def input_frequency(self, value: float) -> None:
        """Set the input frequency. This affects the time horizon for velocity calculation:
        lower frequency = slower response, higher frequency = faster response."""
        if value <= 0.0:
            raise ValueError("Frequency must be positive.")
        self._input_interval = 1.0 / value

    @property
    def responsiveness(self) -> float:
        """Get the current responsiveness (velocity convergence rate)."""
        return self._responsiveness

    @responsiveness.setter
    def responsiveness(self, value: float) -> None:
        """Set the responsiveness (0.0 to 1.0). Higher values make velocity converge faster
        toward the target velocity, resulting in snappier but potentially less smooth motion."""
        self._responsiveness = max(0.0, min(1.0, value))

    @property
    def friction(self) -> float:
        """Get the current friction (velocity damping factor)."""
        return 1.0 - self._inv_friction

    @friction.setter
    def friction(self, value: float) -> None:
        """Set the friction (0.0 to 1.0). Higher values increase velocity damping,
        preventing overshoot and oscillation at the cost of slower response."""
        value = max(0.0, min(1.0, value))
        self._inv_friction = 1.0 - value

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
        """Get the interpolated values for the next frame."""
        return self._interpolated.copy()

    def set_target(self, values: np.ndarray) -> None:
        """Set new target values to interpolate towards.

        This method should be called at the input frequency (e.g., when new pose data arrives).
        The interpolator will smoothly chase these targets using perpetual chase dynamics.

        The algorithm never fully reaches the target before it updates - this is intentional!
        It creates continuous smooth motion without hard stops. Each update() call recalculates
        velocity based on the current distance, creating automatic adaptive deceleration as
        the interpolated value approaches the target.
        """

        if values.shape[0] != self._vector_size:
            raise ValueError(f"Expected array of size {self._vector_size}, got {values.shape[0]}")

        self._target = values

        # Handle newly valid values (including first initialization)
        newly_valid = np.isnan(self._interpolated) & np.isfinite(values)
        if np.any(newly_valid):
            self._interpolated[newly_valid] = values[newly_valid]
            self._v_curr[newly_valid] = 0.0

    def update(self, current_time: float | None = None) -> None:
        """Update the interpolated value for the current time.

        This method should be called at the render frequency (e.g., every display frame).
        It advances the interpolation based on elapsed time using the perpetual chase algorithm:

        1. Calculate target velocity: v_target = (target - current) / input_interval
           This represents "how fast to move to reach target in one input period"

        2. Steer current velocity toward target velocity:
           v_curr += (v_target - v_curr) * responsiveness
           Higher responsiveness = faster velocity convergence

        3. Apply friction (velocity damping):
           v_curr *= (1 - friction)
           Prevents overshoot and oscillation

        4. Update position:
           position += v_curr * dt

        The magic: By recalculating v_target every frame based on current distance,
        the system naturally decelerates as it approaches the target. This creates
        smooth adaptive easing without explicit easing curves.
        """

        if np.all(np.isnan(self._target)):
            return

        if current_time is None:
            current_time = monotonic()

        # Initialize on first update
        if self._last_update_time is None:
            self._last_update_time = current_time
            return

        if current_time <= self._last_update_time:
            if current_time < self._last_update_time:
                warnings.warn("Interpolator received out-of-order time update.", RuntimeWarning)
            return  # Skip update for non-positive dt

        dt: float = current_time - self._last_update_time
        self._last_update_time = current_time

        # Calculate target velocity (PID Proportional term)
        # self._v_target = self._calculate_velocity(self._interpolated, self._target, self._input_interval)


        # Calculate target velocity (PID Proportional term)
        new_v_target = self._calculate_velocity(self._interpolated, self._target, self._input_interval)

        # Smooth the v_target transition to prevent velocity jumps
        self._v_target += (new_v_target - self._v_target) * self._responsiveness

        # Velocity steering (PID Derivative term)
        velocity_correction = (self._v_target - self._v_curr) * self._responsiveness
        self._v_curr = (self._v_curr + velocity_correction) * self._inv_friction

        # Update position
        delta_position = self._v_curr * dt
        self._interpolated = self._interpolated + delta_position

        # Apply constraints (clamping or wrapping)
        self._apply_constraints()

    def _apply_constraints(self) -> None:
        """Apply value constraints (clamping for vectors, overridden for angles)."""
        if self._clamp_range is not None:
            np.clip(self._interpolated, self._clamp_range[0], self._clamp_range[1], out=self._interpolated)

    @staticmethod
    def _calculate_velocity(p_prev: np.ndarray, p_curr: np.ndarray, interval: float) -> np.ndarray:
        """Calculate target velocity from current position to target position."""
        delta = p_curr - p_prev
        v = delta / interval
        return np.nan_to_num(v, nan=0.0)


class AngleChase(VectorChase):
    """Chase interpolator for angular/circular data with proper wrapping."""

    def __init__(self, vector_size: int, input_frequency: float = 30.0, responsiveness: float = 0.2,
                 friction: float = 0.03, clamp_range: tuple[float, float] | None = None) -> None:
        """Initialize the angle chase interpolator."""
        super().__init__(vector_size, input_frequency, responsiveness, friction, clamp_range=None)

    def _apply_constraints(self) -> None:
        """Apply angular wrapping to [-π, π]."""
        self._interpolated = np.arctan2(np.sin(self._interpolated), np.cos(self._interpolated))

    @staticmethod
    def _calculate_velocity(p_prev: np.ndarray, p_curr: np.ndarray, interval: float) -> np.ndarray:
        """Calculate target angular velocity using shortest angular path."""
        delta = p_curr - p_prev
        delta = np.arctan2(np.sin(delta), np.cos(delta))
        v = delta / interval
        return np.nan_to_num(v, nan=0.0)


class PointChase(VectorChase):
    """Chase interpolator for 2D points with (x, y) coordinates."""

    def __init__(self, vector_size: int, input_frequency: float = 30.0, responsiveness: float = 0.2,
                 friction: float = 0.03, clamp_range: tuple[float, float] | None = None) -> None:
        """Initialize the point chase interpolator."""
        super().__init__(vector_size * 2, input_frequency, responsiveness, friction, clamp_range)
        self._num_points: int = vector_size

    def set_target(self, points: np.ndarray) -> None:
        """Set new target points."""
        if points.shape != (self._num_points, 2):
            raise ValueError(f"Expected shape ({self._num_points}, 2), got {points.shape}")
        super().set_target(points.copy().flatten())

    @property
    def value(self) -> np.ndarray:
        """Get the current interpolated points."""
        return self._interpolated.reshape(self._num_points, 2)


Chase = Union[
    VectorChase,
    AngleChase,
    PointChase,
]