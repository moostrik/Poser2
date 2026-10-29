# Standard library imports
from enum import IntEnum

# Third-party imports
import numpy as np


class PredictionMethod(IntEnum):
    NONE = 0
    LINEAR = 1
    QUADRATIC = 2


class Predict:
    """A motion fit on the last samples of a vector and their stamps, evaluated at any time.

    ``add_sample(values, timestamp)`` keeps the newest ``fit_frames`` samples; ``at(time)`` evaluates the fit
    there. LINEAR is the least-squares velocity over the window, through the newest sample; QUADRATIC adds
    the acceleration (three samples or more, else linear); NONE holds the newest sample. The fit is per
    component: a component's window reaches back to its last NaN, a component with one sample holds it, and
    a NaN newest sample gives NaN. Not thread-safe by design."""

    def __init__(self, vector_size: int, method: PredictionMethod, fit_frames: int = 3,
                 clamp_range: tuple[float, float] | None = None) -> None:
        if vector_size <= 0:
            raise ValueError("Vector size must be positive.")
        if clamp_range is not None:
            if len(clamp_range) != 2 or clamp_range[0] >= clamp_range[1]:
                raise ValueError("clamp_range must be (min, max) with min < max")

        self._vector_size: int = vector_size
        self._method: PredictionMethod = method
        self._fit_frames: int = max(2, int(fit_frames))
        self._clamp_range: tuple[float, float] | None = clamp_range

        # The window, newest last: samples (frames x vector) and their stamps.
        self._samples: np.ndarray = np.full((0, vector_size), np.nan)
        self._stamps: np.ndarray = np.zeros(0)
        # The fit through the newest sample: value, velocity and half the acceleration per component.
        self._p: np.ndarray = np.full(vector_size, np.nan)
        self._v: np.ndarray = np.zeros(vector_size)
        self._a: np.ndarray = np.zeros(vector_size)
        self._t: float = 0.0

    @property
    def method(self) -> PredictionMethod:
        return self._method

    @method.setter
    def method(self, value: PredictionMethod) -> None:
        if not isinstance(value, PredictionMethod):
            raise TypeError(f"Expected PredictionMethod, got {type(value).__name__}")
        self._method = value

    @property
    def fit_frames(self) -> int:
        return self._fit_frames

    @fit_frames.setter
    def fit_frames(self, value: int) -> None:
        self._fit_frames = max(2, int(value))

    @property
    def clamp_range(self) -> tuple[float, float] | None:
        return self._clamp_range

    @clamp_range.setter
    def clamp_range(self, value: tuple[float, float] | None) -> None:
        if value is not None:
            if len(value) != 2 or value[0] >= value[1]:
                raise ValueError("clamp_range must be (min, max) with min < max")
        self._clamp_range = value

    @property
    def timestamp(self) -> float:
        """The newest sample's stamp."""
        return self._t

    @property
    def samples(self) -> int:
        """How many samples the window holds."""
        return int(self._samples.shape[0])

    @property
    def stamps(self) -> np.ndarray:
        """The window's stamps, oldest first."""
        return self._stamps

    def coefficients(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """The fit through the newest sample: value, velocity, half the acceleration, and its stamp."""
        return self._p, self._v, self._a, self._t

    def constrain(self, values: np.ndarray) -> np.ndarray:
        """Clamp (or wrap) values the way this fit's outputs are."""
        return self._constrain(values)

    def add_sample(self, values: np.ndarray, timestamp: float) -> None:
        """Add a sample with its stamp and refit."""
        if values.shape[0] != self._vector_size:
            raise ValueError(f"Expected array of size {self._vector_size}, got {values.shape[0]}")
        keep = self._fit_frames - 1
        self._samples = np.vstack((self._samples[-keep:] if keep else self._samples[:0], values[np.newaxis, :]))
        self._stamps = np.append(self._stamps[-keep:] if keep else self._stamps[:0], float(timestamp))
        self._fit()

    def at(self, time: float) -> np.ndarray:
        """The fit evaluated at ``time`` (the same clock as the stamps)."""
        dt = time - self._t
        result = self._p + self._v * dt + self._a * dt * dt
        return self._constrain(result)

    def reset(self) -> None:
        """Forget the window."""
        self._samples = np.full((0, self._vector_size), np.nan)
        self._stamps = np.zeros(0)
        self._p = np.full(self._vector_size, np.nan)
        self._v = np.zeros(self._vector_size)
        self._a = np.zeros(self._vector_size)
        self._t = 0.0

    # -- The fit ---------------------------------------------------------------------

    def _fit(self) -> None:
        newest = self._samples[-1]
        self._t = float(self._stamps[-1])
        self._p = newest.copy()
        self._v = np.zeros(self._vector_size)
        self._a = np.zeros(self._vector_size)
        n = self._samples.shape[0]
        if self._method == PredictionMethod.NONE or n < 2:
            return

        # Per component, the window reaches back from the newest sample to its last NaN.
        valid = np.isfinite(self._samples)                                   # frames x vector
        run = np.flipud(np.cumprod(np.flipud(valid), axis=0)).astype(bool)  # True while valid back from the newest
        weights = run[:-1].astype(float)                                     # the older samples that take part
        t = (self._stamps[:-1] - self._t)[:, np.newaxis]                     # their time before the newest, < 0
        d = self._difference(self._samples[:-1], newest)                     # their value relative to the newest
        d = np.where(run[:-1], d, 0.0)

        # Least squares through the newest sample: d = v t (+ a t²), weighted by the window.
        st2 = np.sum(weights * t * t, axis=0)
        std = np.sum(weights * t * d, axis=0)
        if self._method == PredictionMethod.QUADRATIC:
            st3 = np.sum(weights * t ** 3, axis=0)
            st4 = np.sum(weights * t ** 4, axis=0)
            st2d = np.sum(weights * t * t * d, axis=0)
            det = st2 * st4 - st3 * st3
            quadratic = (np.sum(weights, axis=0) >= 2) & (np.abs(det) > 1e-18)
            with np.errstate(divide='ignore', invalid='ignore'):
                v_q = (std * st4 - st2d * st3) / det
                a_q = (st2d * st2 - std * st3) / det
                v_l = std / st2
            self._v = np.where(quadratic, v_q, np.where(st2 > 0.0, v_l, 0.0))
            self._a = np.where(quadratic, a_q, 0.0)
        else:
            with np.errstate(divide='ignore', invalid='ignore'):
                self._v = np.where(st2 > 0.0, std / st2, 0.0)
        self._v = np.nan_to_num(self._v, nan=0.0, posinf=0.0, neginf=0.0)
        self._a = np.nan_to_num(self._a, nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def _difference(older: np.ndarray, newest: np.ndarray) -> np.ndarray:
        """The older samples relative to the newest."""
        return older - newest[np.newaxis, :]

    def _constrain(self, values: np.ndarray) -> np.ndarray:
        if self._clamp_range is not None:
            np.clip(values, self._clamp_range[0], self._clamp_range[1], out=values)
        return values


class AnglePredict(Predict):
    """The fit for angles: differences take the short way round and the result wraps to [-π, π]."""

    def __init__(self, vector_size: int, method: PredictionMethod, fit_frames: int = 3,
                 clamp_range: tuple[float, float] | None = None) -> None:
        super().__init__(vector_size, method, fit_frames, clamp_range=None)

    @staticmethod
    def _difference(older: np.ndarray, newest: np.ndarray) -> np.ndarray:
        delta = older - newest[np.newaxis, :]
        return np.arctan2(np.sin(delta), np.cos(delta))

    def _constrain(self, values: np.ndarray) -> np.ndarray:
        return np.arctan2(np.sin(values), np.cos(values))


class PointPredict(Predict):
    """The fit for 2D points, (x, y) flattened."""

    def __init__(self, vector_size: int, method: PredictionMethod, fit_frames: int = 3,
                 clamp_range: tuple[float, float] | None = None) -> None:
        super().__init__(vector_size * 2, method, fit_frames, clamp_range)
        self._num_points: int = vector_size

    def add_sample(self, points: np.ndarray, timestamp: float) -> None:
        if points.shape != (self._num_points, 2):
            raise ValueError(f"Expected shape ({self._num_points}, 2), got {points.shape}")
        super().add_sample(points.flatten(), timestamp)

    def at(self, time: float) -> np.ndarray:
        return super().at(time).reshape(self._num_points, 2)


ArrayPredict = Predict | AnglePredict | PointPredict
