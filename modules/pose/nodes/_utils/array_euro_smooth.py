# Standard library imports
import math
from typing import Union

# Third-party imports
import numpy as np


class EuroSmooth:
    """Smoother for arbitrary vector data (positions, coordinates, etc.).

    The One Euro recurrence (Casiez et al.), vectorized over the whole vector:

        dx     = (x - smoothed) * frequency
        edx    = low-pass(dx, d_cutoff)
        cutoff = min_cutoff + cutoff_rise * |edx|
        x_hat  = low-pass(x, cutoff)        with alpha(c) = 1 / (1 + frequency / (2π c))

    Vectorized in numpy rather than one filter object per component: the per-component
    Python loop dominated the pose chain's GIL time. A component's first valid sample
    passes through unfiltered with its derivative state cleared, so a track starts (and
    reappears after a NaN gap) fresh."""

    def __init__(self, vector_size: int, frequency: float, min_cutoff: float, cutoff_rise: float,
                 d_cutoff: float, clamp_range: tuple[float, float] | None = None) -> None:
        """Initialize the vectorized smoother."""
        if vector_size <= 0:
            raise ValueError("Vector size must be positive.")
        if frequency <= 0.0:
            raise ValueError("Frequency must be positive.")
        if min_cutoff < 0.0:
            raise ValueError("min_cutoff must be non-negative.")
        if cutoff_rise < 0.0:
            raise ValueError("cutoff_rise must be non-negative.")
        if d_cutoff < 0.0:
            raise ValueError("d_cutoff must be non-negative.")
        if clamp_range is not None:
            if len(clamp_range) != 2 or clamp_range[0] >= clamp_range[1]:
                raise ValueError("clamp_range must be (min, max) with min < max")

        self._vector_size: int = vector_size
        self._frequency: float = frequency
        self._min_cutoff: float = min_cutoff
        self._cutoff_rise: float = cutoff_rise
        self._d_cutoff: float = d_cutoff
        self._clamp_range: tuple[float, float] | None = clamp_range

        # Current smoothed values (NaN = no state) and the filtered derivative per component
        self._smoothed: np.ndarray = np.full(vector_size, np.nan)
        self._dx: np.ndarray = np.zeros(vector_size)

    @staticmethod
    def _alpha(frequency: float, cutoff) -> np.ndarray | float:
        """The low-pass alpha for a cutoff (scalar or per-component array)."""
        return 1.0 / (1.0 + frequency / (2.0 * math.pi * cutoff))

    def add_sample(self, values: np.ndarray) -> None:
        """Add a new sample and calculate smoothing.

        NaN handling per component: a NaN input stays NaN in the output and clears that
        component's state; the first valid sample after (or ever) passes through unfiltered
        and starts the recurrence fresh.
        """
        if values.shape[0] != self._vector_size:
            raise ValueError(f"Expected array of size {self._vector_size}, got {values.shape[0]}")

        was_valid = np.isfinite(self._smoothed)
        is_valid = np.isfinite(values)
        both = was_valid & is_valid
        fresh = is_valid & ~was_valid

        # The recurrence over every lane; lanes where it is NaN are discarded by the masks below.
        freq = self._frequency
        dx = (values - self._smoothed) * freq
        a_d = self._alpha(freq, self._d_cutoff)
        edx = a_d * dx + (1.0 - a_d) * self._dx
        cutoff = self._min_cutoff + self._cutoff_rise * np.abs(edx)
        a = self._alpha(freq, np.maximum(cutoff, 1e-12))
        filtered = a * values + (1.0 - a) * self._smoothed

        self._dx = np.where(both, edx, 0.0)
        self._smoothed = np.where(both, filtered, np.where(fresh, values, np.nan))

        self._apply_constraints()

    def reset(self) -> None:
        """Reset the vector and smooth filters."""
        self._smoothed = np.full(self._vector_size, np.nan)
        self._dx = np.zeros(self._vector_size)

    def _apply_constraints(self) -> None:
        """Apply value constraints (clamping for vectors, overridden for angles)."""
        if self._clamp_range is not None:
            np.clip(self._smoothed, self._clamp_range[0], self._clamp_range[1], out=self._smoothed)

    @property
    def value(self) -> np.ndarray:
        """Get the current smoothed values (returns a copy)."""
        return self._smoothed.copy()

    @property
    def frequency(self) -> float:
        """Get the frequency."""
        return self._frequency

    @frequency.setter
    def frequency(self, value: float) -> None:
        """Set the frequency."""
        if value <= 0.0:
            raise ValueError("Frequency must be positive.")
        self._frequency = value

    @property
    def min_cutoff(self) -> float:
        """Get the min cutoff frequency."""
        return self._min_cutoff

    @min_cutoff.setter
    def min_cutoff(self, value: float) -> None:
        """Set the min cutoff frequency."""
        if value < 0.0:
            raise ValueError("min_cutoff must be non-negative.")
        self._min_cutoff = value

    @property
    def cutoff_rise(self) -> float:
        """Get the cutoff rise (the One Euro beta)."""
        return self._cutoff_rise

    @cutoff_rise.setter
    def cutoff_rise(self, value: float) -> None:
        """Set the cutoff rise (the One Euro beta)."""
        if value < 0.0:
            raise ValueError("cutoff_rise must be non-negative.")
        self._cutoff_rise = value

    @property
    def d_cutoff(self) -> float:
        """Get the derivative cutoff frequency."""
        return self._d_cutoff

    @d_cutoff.setter
    def d_cutoff(self, value: float) -> None:
        """Set the derivative cutoff frequency."""
        if value < 0.0:
            raise ValueError("d_cutoff must be non-negative.")
        self._d_cutoff = value

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


class AngleEuroSmooth(EuroSmooth):
    """Smoother for angular/circular data with proper wrapping.

    Filters angles by decomposing into sin/cos components, filtering each
    separately, then reconstructing the angle. This avoids discontinuities
    at the ±π boundary without requiring custom angular filters.
    """

    def __init__(self, vector_size: int, frequency: float, min_cutoff: float, cutoff_rise: float,
                 d_cutoff: float, clamp_range: tuple[float, float] | None = None) -> None:
        """Initialize the angle smoother.

        Note: clamp_range is not supported for angles (automatic wrapping to [-π, π]).
        """
        # Create 2x filters (one for sin, one for cos per angle)
        super().__init__(vector_size * 2, frequency, min_cutoff, cutoff_rise, d_cutoff, clamp_range=None)
        self._num_angles: int = vector_size

    def add_sample(self, angles: np.ndarray) -> None:
        """Add new angle samples.

        Args:
            angles: Angles in radians, shape (num_angles,)
        """
        if angles.shape[0] != self._num_angles:
            raise ValueError(f"Expected {self._num_angles} angles, got {angles.shape[0]}")

        # Decompose angles into sin/cos components
        sin_cos = np.empty(self._num_angles * 2)
        sin_cos[0::2] = np.sin(angles)  # sin components at even indices
        sin_cos[1::2] = np.cos(angles)  # cos components at odd indices

        # Filter sin/cos components using parent implementation
        super().add_sample(sin_cos)

    @property
    def value(self) -> np.ndarray:
        """Get smoothed angles reconstructed from filtered sin/cos components."""
        filtered_sin_cos = super().value

        # Extract sin/cos components
        filtered_sin = filtered_sin_cos[0::2]
        filtered_cos = filtered_sin_cos[1::2]

        # Reconstruct angles
        return np.arctan2(filtered_sin, filtered_cos)

    def _apply_constraints(self) -> None:
        """No clamping for sin/cos components (they're naturally bounded to [-1, 1])."""
        pass


class PointEuroSmooth(EuroSmooth):
    """Smoother for 2D points with (x, y) coordinates."""

    def __init__(self, vector_size: int, frequency: float, min_cutoff: float, cutoff_rise: float,
                 d_cutoff: float, clamp_range: tuple[float, float] | None = None) -> None:
        """Initialize the point smoother."""
        super().__init__(vector_size * 2, frequency, min_cutoff, cutoff_rise, d_cutoff, clamp_range)
        self._num_points: int = vector_size

    def add_sample(self, points: np.ndarray) -> None:
        """Add new point samples with shape (num_points, 2)."""
        if points.shape != (self._num_points, 2):
            raise ValueError(f"Expected shape ({self._num_points}, 2), got {points.shape}")
        super().add_sample(points.flatten())

    @property
    def value(self) -> np.ndarray:
        """Get smoothed points with shape (num_points, 2)."""
        return super().value.reshape(self._num_points, 2)


ArraySmooth = Union[EuroSmooth, AngleEuroSmooth, PointEuroSmooth]