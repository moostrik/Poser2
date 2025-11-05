# Standard library imports
from time import time

# Third-party imports
import numpy as np

# Local application imports
from modules.utils.HotReloadMethods import HotReloadMethods


class VectorAngle:

    def __init__(self, input_rate: float, vector_size: int, responsiveness: float = 0.2, friction: float = 0.03) -> None:
        """Initialize the vectorized angle interpolator with velocity steering."""
        self.input_interval: float = 1.0 / input_rate
        self.vector_size: int = vector_size

        self.responsiveness: float = responsiveness
        self.damping: float = 1.0 - friction

        self._initialized = False

        self.last_sample_time: float = 0.0
        self.last_update_time: float = 0.0

        self.a_prev: np.ndarray = np.full(vector_size, np.nan)
        self.a_curr: np.ndarray = np.full(vector_size, np.nan)
        self.a_target: np.ndarray = np.full(vector_size, np.nan)
        self.a_interpolated: np.ndarray = np.full(vector_size, np.nan)

        self.v_curr: np.ndarray = np.zeros(vector_size)
        self.v_target: np.ndarray = np.zeros(vector_size)


        self._hot_reload = HotReloadMethods(self.__class__, True, True)

    def get_interpolated(self) -> np.ndarray:
        """Get the current interpolated angle values."""
        return self.a_interpolated

    def add_sample(self, angles: np.ndarray, sample_time: float | None = None) -> None:
        """Add a new sample to the interpolator."""

        if angles.shape[0] != self.vector_size:
            raise ValueError(f"Expected array of size {self.vector_size}, got {angles.shape[0]}")

        if sample_time is None:
            sample_time = time()

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

        v_measured: np.ndarray = self._calculate_velocity(self.a_prev, self.a_curr, self.input_interval)
        self.a_target = self._predict_linear(self.a_curr, v_measured, self.input_interval)
        # self.a_target = self.a_curr
        self.v_target = self._calculate_velocity(self.a_interpolated, self.a_target, self.input_interval)


    def update(self, current_time: float | None = None) -> None:
        """Update the interpolated value for the current time."""
        if not self._initialized:
            return

        if current_time is None:
            current_time = time()

        dt: float = current_time - self.last_update_time
        self.last_update_time = current_time


        velocity_correction = (self.v_target - self.v_curr) * self.responsiveness
        self.v_curr = (self.v_curr + velocity_correction) * self.damping
        delta_position = self.v_curr * dt

        self.a_interpolated = self.a_interpolated + delta_position
        self.a_interpolated = np.arctan2(np.sin(self.a_interpolated), np.cos(self.a_interpolated))

        self.responsiveness = 0.2
        self.damping = 1.0 # 0.97

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
