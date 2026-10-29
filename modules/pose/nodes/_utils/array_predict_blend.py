"""The fit of a vector's last frames evaluated at the time it is shown, a new frame's correction eased in.

``set_target(values, timestamp)`` adds a frame to the fit (``array_predict.Predict``) and keeps the fit it replaces;
``at(time)`` evaluates both there and returns the new fit's value with the previous fit's weighted in by the
blend mode and the time since the first evaluation after the frame: at that first evaluation the output equals
what the previous fit showed, so a frame never jumps. The distance past the newest stamp is capped at
``max_horizon``. Not thread-safe by design.
"""

# Standard library imports
from enum import IntEnum
import math

# Third-party imports
import numpy as np

from .array_predict import AnglePredict, PointPredict, Predict, PredictionMethod


class BlendMode(IntEnum):
    """How a new frame's correction is eased in."""
    NONE   = 0      # lands at once: a step at every frame
    RAMP   = 1      # a straight line, lands after `blend` seconds
    DAMPEN = 2      # exponential, `blend` its time constant: soft, no overshoot, fades out


DEFAULT_INTERVAL: float = 1.0 / 30.0     # the blend's span before the stamps say the camera's interval


class PredictBlend:
    """The fit evaluated at any time, with the previous fit blended in after a frame."""

    def __init__(self, vector_size: int, method: PredictionMethod, fit_frames: int, blend_mode: BlendMode,
                 blend: float, max_horizon: float, clamp_range: tuple[float, float] | None = None) -> None:
        self._vector_size: int = vector_size
        self._fit: Predict = self._make_fit(vector_size, method, fit_frames, clamp_range)
        self.blend_mode: BlendMode = blend_mode
        self.blend: float = blend                      # seconds; 0 = one camera interval, from the stamps
        self.max_horizon: float = max_horizon
        # The previous fit's coefficients (None before the second frame) and the time of the first
        # evaluation after the newest frame (None until it happens)
        self._previous: tuple[np.ndarray, np.ndarray, np.ndarray, float] | None = None
        self._blend_from: float | None = None
        self._interval: float = DEFAULT_INTERVAL
        self.horizon: float = 0.0                      # the last evaluation's distance past the newest stamp (s)

    @staticmethod
    def _make_fit(vector_size: int, method: PredictionMethod, fit_frames: int,
                  clamp_range: tuple[float, float] | None) -> Predict:
        return Predict(vector_size, method, fit_frames, clamp_range)

    @property
    def fit(self) -> Predict:
        return self._fit

    def set_target(self, values: np.ndarray, timestamp: float) -> None:
        """Add the frame to the fit and keep the fit it replaces."""
        if values.shape[0] != self._vector_size:
            raise ValueError(f"Expected array of size {self._vector_size}, got {values.shape[0]}")
        if self._fit.samples > 0:
            self._previous = self._fit.coefficients()
            stamps = self._fit.stamps
            if stamps.shape[0] >= 2:
                self._interval = float(stamps[-1] - stamps[0]) / (stamps.shape[0] - 1)
        self._fit.add_sample(values, timestamp)
        self._blend_from = None

    def at(self, time: float) -> np.ndarray:
        """The value shown at ``time``."""
        value = self._evaluate(self._fit.coefficients(), time)
        if self._previous is None or self.blend_mode == BlendMode.NONE:
            return self._constrain(value)
        if self._blend_from is None:
            self._blend_from = time
        weight = self._weight(time - self._blend_from)
        if weight <= 0.0:
            self._previous = None
            return self._constrain(value)
        old = self._evaluate(self._previous, time)
        # Where the previous fit has no value, the new one stands alone
        blended = np.where(np.isfinite(old), value + self._difference(old, value) * weight, value)
        return self._constrain(blended)

    def reset(self) -> None:
        self._fit.reset()
        self._previous = None
        self._blend_from = None
        self._interval = DEFAULT_INTERVAL
        self.horizon = 0.0

    # -- Internals -----------------------------------------------------------------------

    def _evaluate(self, coefficients: tuple[np.ndarray, np.ndarray, np.ndarray, float], time: float) -> np.ndarray:
        p, v, a, t = coefficients
        dt = min(time - t, self.max_horizon)
        if coefficients[3] == self._fit.timestamp:
            self.horizon = dt
        return p + v * dt + a * dt * dt

    def _weight(self, elapsed: float) -> float:
        """The previous fit's share, ``elapsed`` seconds after the first evaluation past the frame."""
        span = self.blend if self.blend > 0.0 else self._interval
        if span <= 0.0:
            return 0.0
        if self.blend_mode == BlendMode.RAMP:
            return max(0.0, 1.0 - elapsed / span)
        weight = math.exp(-elapsed / span)
        return weight if weight > 1e-3 else 0.0

    @staticmethod
    def _difference(old: np.ndarray, new: np.ndarray) -> np.ndarray:
        return old - new

    def _constrain(self, values: np.ndarray) -> np.ndarray:
        return self._fit.constrain(values)


class AnglePredictBlend(PredictBlend):
    """For angles: the correction takes the short way round and the result wraps."""

    @staticmethod
    def _make_fit(vector_size: int, method: PredictionMethod, fit_frames: int,
                  clamp_range: tuple[float, float] | None) -> Predict:
        return AnglePredict(vector_size, method, fit_frames)

    @staticmethod
    def _difference(old: np.ndarray, new: np.ndarray) -> np.ndarray:
        delta = old - new
        return np.arctan2(np.sin(delta), np.cos(delta))


class PointPredictBlend(PredictBlend):
    """For 2D points, (x, y) flattened."""

    def __init__(self, vector_size: int, method: PredictionMethod, fit_frames: int, blend_mode: BlendMode,
                 blend: float, max_horizon: float, clamp_range: tuple[float, float] | None = None) -> None:
        super().__init__(vector_size * 2, method, fit_frames, blend_mode, blend, max_horizon, clamp_range)
        self._num_points: int = vector_size

    def set_target(self, points: np.ndarray, timestamp: float) -> None:
        if points.shape != (self._num_points, 2):
            raise ValueError(f"Expected shape ({self._num_points}, 2), got {points.shape}")
        super().set_target(points.flatten(), timestamp)

    def at(self, time: float) -> np.ndarray:
        return super().at(time).reshape(self._num_points, 2)
