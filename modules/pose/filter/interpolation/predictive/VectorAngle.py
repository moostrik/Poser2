"""
Feel	responsiveness	friction	Result
Snappy	0.4-0.6	        0.05-0.08	Quick response, slight overshoot
Smooth	0.15-0.25	    0.02-0.04	Slower, fluid motion
Tight	0.3-0.4	        0.03-0.05	Balanced, no lag
Floaty	0.1-0.15	    0.01-0.02	Slow, dreamy motion
"""


# Standard library imports
from time import monotonic

# Third-party imports
import numpy as np
import warnings

# Local application imports
from modules.utils.HotReloadMethods import HotReloadMethods


# 2DO
# monotmoc time from PoseInterpolator
# make other predictions

class VectorAngle:

    def __init__(self, input_rate: float, vector_size: int, responsiveness: float = 0.2, friction: float = 0.03) -> None:
        """Initialize the vectorized angle interpolator with velocity steering."""
        self.input_interval: float = 1.0 / input_rate
        self.vector_size: int = vector_size

        self.responsiveness: float = responsiveness
        self.inv_friction: float = 1.0 - friction

        self._initialized = False

        self.last_sample_time: float = 0.0
        self.last_update_time: float = 0.0

        self.a_prev: np.ndarray = np.full(vector_size, np.nan)
        self.a_curr: np.ndarray = np.full(vector_size, np.nan)
        self.a_target: np.ndarray = np.full(vector_size, np.nan)
        self.a_interpolated: np.ndarray = np.full(vector_size, np.nan)

        self.v_prev: np.ndarray = np.zeros(vector_size)
        self.v_curr: np.ndarray = np.zeros(vector_size)
        self.v_target: np.ndarray = np.zeros(vector_size)


        self._hot_reload = HotReloadMethods(self.__class__, True, True)

    def get_interpolated(self) -> np.ndarray:
        """Get the current interpolated angle values."""
        return self.a_interpolated.copy()

    def add_sample(self, angles: np.ndarray, sample_time: float | None = None) -> None:
        """Add a new sample to the interpolator."""

        if angles.shape[0] != self.vector_size:
            raise ValueError(f"Expected array of size {self.vector_size}, got {angles.shape[0]}")

        if sample_time is None:
            sample_time = monotonic()

        # Shift samples
        self.a_prev = self.a_curr
        self.a_curr = angles
        self.last_sample_time = sample_time

        # Initialize on first sample
        if not self._initialized:
            self.a_interpolated = angles.copy()
            self.last_update_time = sample_time
            self._initialized = True
            return

        newly_valid = np.isnan(self.a_interpolated) & np.isfinite(angles)
        if np.any(newly_valid):
            self.a_interpolated[newly_valid] = angles[newly_valid]
            self.v_curr[newly_valid] = 0.0
            self.v_prev[newly_valid] = 0.0

        v_measured: np.ndarray = self._calculate_velocity(self.a_prev, self.a_curr, self.input_interval)
        acceleration: np.ndarray = self._calculate_velocity(self.v_prev, v_measured, self.input_interval)
        self.v_prev = v_measured

        self.a_target = self._predict_quadratic(self.a_curr, v_measured, acceleration, self.input_interval)
        # self.a_target = self._predict_linear(self.a_curr, v_measured, self.input_interval)

    def update(self, current_time: float | None = None) -> None:
        """Update the interpolated value for the current time."""
        if not self._initialized:
            return

        if current_time is None:
            current_time = monotonic()

        if current_time < self.last_update_time:
            warnings.warn("Interpolator received out-of-order time update.", RuntimeWarning)
            current_time = self.last_update_time

        dt: float = current_time - self.last_update_time
        dt = min(dt, 0.1)  # Max 100ms step
        self.last_update_time = current_time

        self.v_target = self._calculate_velocity(self.a_interpolated, self.a_target, self.input_interval)

        velocity_correction = (self.v_target - self.v_curr) * self.responsiveness
        self.v_curr = (self.v_curr + velocity_correction) * self.inv_friction
        delta_position = self.v_curr * dt

        self.a_interpolated = self.a_interpolated + delta_position
        self.a_interpolated = np.arctan2(np.sin(self.a_interpolated), np.cos(self.a_interpolated))

        # self.responsiveness = 0.2
        # self.damping = 0.97

    @staticmethod
    def _calculate_velocity(p_prev: np.ndarray, p_curr: np.ndarray, interval: float) -> np.ndarray:
        """Calculate instantaneous angular velocity from two samples."""
        # Calculate angular differences using shortest path
        delta = p_curr - p_prev
        delta: np.ndarray = np.arctan2(np.sin(delta), np.cos(delta))
        v = delta / interval

        # Replace NaN with zero
        return np.nan_to_num(v, nan=0.0)

    @staticmethod
    def _predict_linear(p_curr: np.ndarray, v_curr: np.ndarray, interval: float) -> np.ndarray:
        """Predict next angle position based on current velocity."""
        p_pred_raw = p_curr + v_curr * interval
        return np.arctan2(np.sin(p_pred_raw), np.cos(p_pred_raw))

    @staticmethod
    def _predict_quadratic(p_curr: np.ndarray, v_curr: np.ndarray, accel: np.ndarray, interval: float) -> np.ndarray:
        """Predict next angle position using quadratic extrapolation (constant acceleration model)."""
        # Formula: p(t) = p₀ + v₀*t + ½*a*t²
        p_pred_raw = p_curr + v_curr * interval + 0.5 * accel * (interval ** 2)
        return np.arctan2(np.sin(p_pred_raw), np.cos(p_pred_raw))
