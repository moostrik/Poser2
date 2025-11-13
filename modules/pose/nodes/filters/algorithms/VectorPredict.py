# Standard library imports
from enum import IntEnum
from typing import Union

# Third-party imports
import numpy as np

class PredictionMethod(IntEnum):
    NONE = 0
    LINEAR = 1
    QUADRATIC = 2

class VectorPredict:
    """Predictor for arbitrary vector data (positions, coordinates, etc.)."""

    def __init__(self, vector_size: int, input_frequency: float, method: PredictionMethod,
                 clamp_range: tuple[float, float] | None = None) -> None:
        """Initialize the vectorized predictor."""

        if vector_size <= 0:
            raise ValueError("Vector size must be positive.")
        if input_frequency <= 0.0:
            raise ValueError("Frequency must be positive.")
        if clamp_range is not None:
            if len(clamp_range) != 2 or clamp_range[0] >= clamp_range[1]:
                raise ValueError("clamp_range must be (min, max) with min < max")

        self._vector_size: int = vector_size
        self._input_interval: float = 1.0 / input_frequency
        self._method: PredictionMethod = method
        self._clamp_range: tuple[float, float] | None = clamp_range

        self.p_prev: np.ndarray = np.full(vector_size, np.nan)
        self.p_curr: np.ndarray = np.full(vector_size, np.nan)
        self.p_predicted: np.ndarray = np.full(vector_size, np.nan)

        self.v_prev: np.ndarray = np.zeros(vector_size)

    @property
    def input_frequency(self) -> float:
        """Get the input frequency."""
        return 1.0 / self._input_interval

    @input_frequency.setter
    def input_frequency(self, value: float) -> None:
        """Set the input frequency."""
        if value <= 0.0:
            raise ValueError("Frequency must be positive.")
        self._input_interval = 1.0 / value

    @property
    def method(self) -> PredictionMethod:
        """Get the current prediction method."""
        return self._method

    @method.setter
    def method(self, value: PredictionMethod) -> None:
        """Set the prediction method."""
        if not isinstance(value, PredictionMethod):
            raise TypeError(f"Expected PredictionMethod, got {type(value).__name__}")
        self._method = value

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
        """Get the predicted values for the next frame."""
        return self.p_predicted.copy()

    def add_sample(self, values: np.ndarray) -> None:
        """Add a new sample and calculate prediction."""

        if values.shape[0] != self._vector_size:
            raise ValueError(f"Expected array of size {self._vector_size}, got {values.shape[0]}")

        # Shift samples
        self.p_prev = self.p_curr
        self.p_curr = values

        newly_valid = np.isnan(self.p_prev) & np.isfinite(self.p_curr)
        if np.any(newly_valid):
            self.v_prev[newly_valid] = 0.0

        # Calculate velocity and acceleration
        v_measured: np.ndarray = self._calculate_velocity(self.p_prev, self.p_curr, self._input_interval)

        # Predict next position
        if self._method == PredictionMethod.NONE:
            self.p_predicted = self.p_curr.copy()
        elif self._method == PredictionMethod.LINEAR:
            self.p_predicted = self._predict_linear(self.p_curr, v_measured, self._input_interval)
        else:  # QUADRATIC
            acceleration: np.ndarray = self._calculate_velocity(self.v_prev, v_measured, self._input_interval)
            self.p_predicted = self._predict_quadratic(self.p_curr, v_measured, acceleration, self._input_interval)

        self.v_prev = v_measured

        if self._clamp_range is not None:
            np.clip(self.p_predicted, self._clamp_range[0], self._clamp_range[1], out=self.p_predicted)

    def reset(self) -> None:
        """Reset the predictor's internal state (clear sample history)."""
        self.p_prev = np.full(self._vector_size, np.nan)
        self.p_curr = np.full(self._vector_size, np.nan)
        self.p_predicted = np.full(self._vector_size, np.nan)
        self.v_prev = np.zeros(self._vector_size)

    @staticmethod
    def _calculate_velocity(p_prev: np.ndarray, p_curr: np.ndarray, interval: float) -> np.ndarray:
        """Calculate instantaneous velocity from two samples."""
        delta = p_curr - p_prev
        v = delta / interval
        return np.nan_to_num(v, nan=0.0)

    @staticmethod
    def _predict_linear(p_curr: np.ndarray, v_curr: np.ndarray, interval: float) -> np.ndarray:
        """Predict next position based on current velocity."""
        return p_curr + v_curr * interval

    @staticmethod
    def _predict_quadratic(p_curr: np.ndarray, v_curr: np.ndarray, accel: np.ndarray, interval: float) -> np.ndarray:
        """Predict next position using quadratic extrapolation (constant acceleration model)."""
        return p_curr + v_curr * interval + 0.5 * accel * (interval ** 2)


class AnglePredict(VectorPredict):
    """Predictor for angular/circular data with proper wrapping."""

    def __init__(self, vector_size: int, input_frequency: float, method: PredictionMethod,
                 clamp_range: tuple[float, float] | None = None) -> None:
        """Initialize the angle predictor.

        Note: clamp_range is not supported for angles (automatic wrapping to [-π, π]).
        """
        super().__init__(vector_size, input_frequency, method, clamp_range=None)

    @staticmethod
    def _calculate_velocity(p_prev: np.ndarray, p_curr: np.ndarray, interval: float) -> np.ndarray:
        """Calculate instantaneous angular velocity from two samples."""
        delta = p_curr - p_prev
        delta = np.arctan2(np.sin(delta), np.cos(delta))  # Wrap to shortest path
        v = delta / interval
        return np.nan_to_num(v, nan=0.0)

    @staticmethod
    def _predict_linear(p_curr: np.ndarray, v_curr: np.ndarray, interval: float) -> np.ndarray:
        """Predict next angle position based on current velocity."""
        p_pred_raw = p_curr + v_curr * interval
        return np.arctan2(np.sin(p_pred_raw), np.cos(p_pred_raw))  # Wrap to [-π, π]

    @staticmethod
    def _predict_quadratic(p_curr: np.ndarray, v_curr: np.ndarray, accel: np.ndarray, interval: float) -> np.ndarray:
        """Predict next angle position using quadratic extrapolation (constant acceleration model)."""
        p_pred_raw = p_curr + v_curr * interval + 0.5 * accel * (interval ** 2)
        return np.arctan2(np.sin(p_pred_raw), np.cos(p_pred_raw))  # Wrap to [-π, π]


class PointPredict(VectorPredict):
    """Predictor for 2D points with (x, y) coordinates."""

    def __init__(self, vector_size: int, input_frequency: float, method: PredictionMethod,
                 clamp_range: tuple[float, float] | None = None) -> None:
        """Initialize the point predictor."""
        super().__init__(vector_size * 2, input_frequency, method, clamp_range)
        self._num_points: int = vector_size

    def add_sample(self, points: np.ndarray) -> None:
        """Add new point samples."""
        if points.shape != (self._num_points, 2):
            raise ValueError(f"Expected shape ({self._num_points}, 2), got {points.shape}")
        super().add_sample(points.flatten())

    @property
    def value(self) -> np.ndarray:
        """Get predicted points for the next frame."""
        return super().value.reshape(self._num_points, 2)

Predict = Union[
    VectorPredict,
    AnglePredict,
    PointPredict,
]