# Standard library imports
from typing import Union

# Third-party imports
import numpy as np
from OneEuroFilter import OneEuroFilter

# Local application imports
from modules.utils.HotReloadMethods import HotReloadMethods


class VectorSmooth:
    """Smoother for arbitrary vector data (positions, coordinates, etc.)."""

    def __init__(self, vector_size: int, frequency: float, min_cutoff: float, beta: float,
                 d_cutoff: float, clamp_range: tuple[float, float] | None = None) -> None:
        """Initialize the vectorized smoother."""
        if vector_size <= 0:
            raise ValueError("Vector size must be positive.")
        if frequency <= 0.0:
            raise ValueError("Frequency must be positive.")
        if min_cutoff < 0.0:
            raise ValueError("min_cutoff must be non-negative.")
        if beta < 0.0:
            raise ValueError("beta must be non-negative.")
        if d_cutoff < 0.0:
            raise ValueError("d_cutoff must be non-negative.")
        if clamp_range is not None:
            if len(clamp_range) != 2 or clamp_range[0] >= clamp_range[1]:
                raise ValueError("clamp_range must be (min, max) with min < max")

        self._vector_size: int = vector_size
        self._frequency: float = frequency
        self._min_cutoff: float = min_cutoff
        self._beta: float = beta
        self._d_cutoff: float = d_cutoff
        self._clamp_range: tuple[float, float] | None = clamp_range

        # Current smoothed values (initialized to NaN)
        self._smoothed: np.ndarray = np.full(vector_size, np.nan)

        # Create OneEuroFilters for each vector component
        self._filters: list[OneEuroFilter] = []
        self._create_filters()

        self._hot_reload = HotReloadMethods(self.__class__, True, True)

    def add_sample(self, values: np.ndarray) -> None:
        """Add a new sample and calculate smoothing.

        OneEuroFilter does not handle NaN inputs gracefully - NaN corrupts
        the internal state. We must handle NaN explicitly:
        - Skip filtering for NaN inputs (preserve NaN in output)
        - Reset filter when transitioning from NaN to valid
        - Reset filter when transitioning from valid to NaN (to clear corrupted state)
        """
        if values.shape[0] != self._vector_size:
            raise ValueError(f"Expected array of size {self._vector_size}, got {values.shape[0]}")

        # Compute validity masks once
        was_valid = np.isfinite(self._smoothed)
        is_valid = np.isfinite(values)

        for i in range(self._vector_size):
            if not is_valid[i]:
                # Input is NaN → output NaN, reset filter if was valid
                if was_valid[i]:
                    self._filters[i].reset()
                self._smoothed[i] = np.nan
            elif not was_valid[i]:
                # NaN → Valid transition: reset filter and initialize
                self._filters[i].reset()
                self._smoothed[i] = values[i]
            else:
                # Both valid → filter normally
                self._smoothed[i] = self._filters[i](values[i])

        self._apply_constraints()

    def reset(self) -> None:
        """Reset the vector and smooth filters."""
        self._smoothed.fill(np.nan)
        for filter in self._filters:
            filter.reset()

    def _create_filters(self) -> None:
        """Create the OneEuroFilters for each vector component."""
        self._filters = [
            OneEuroFilter(self._frequency, self._min_cutoff, self._beta, self._d_cutoff)
            for _ in range(self._vector_size)
        ]

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
        for filter in self._filters:
            filter.setFrequency(value)

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
        for filter in self._filters:
            filter.setMinCutoff(value)

    @property
    def beta(self) -> float:
        """Get the beta parameter."""
        return self._beta

    @beta.setter
    def beta(self, value: float) -> None:
        """Set the beta parameter."""
        if value < 0.0:
            raise ValueError("beta must be non-negative.")
        self._beta = value
        for filter in self._filters:
            filter.setBeta(value)

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
        for filter in self._filters:
            filter.setDerivateCutoff(value)

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


class AngleSmooth(VectorSmooth):
    """Smoother for angular/circular data with proper wrapping.

    Filters angles by decomposing into sin/cos components, filtering each
    separately, then reconstructing the angle. This avoids discontinuities
    at the ±π boundary without requiring custom angular filters.
    """

    def __init__(self, vector_size: int, frequency: float, min_cutoff: float, beta: float,
                 d_cutoff: float) -> None:
        """Initialize the angle smoother.

        Note: clamp_range is not supported for angles (automatic wrapping to [-π, π]).
        """
        # Create 2x filters (one for sin, one for cos per angle)
        super().__init__(vector_size * 2, frequency, min_cutoff, beta, d_cutoff, clamp_range=None)
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


class PointSmooth(VectorSmooth):
    """Smoother for 2D points with (x, y) coordinates."""

    def __init__(self, num_points: int, frequency: float, min_cutoff: float, beta: float,
                 d_cutoff: float, clamp_range: tuple[float, float] | None = None) -> None:
        """Initialize the point smoother."""
        super().__init__(num_points * 2, frequency, min_cutoff, beta, d_cutoff, clamp_range)
        self._num_points: int = num_points

    def add_sample(self, points: np.ndarray) -> None:
        """Add new point samples with shape (num_points, 2)."""
        if points.shape != (self._num_points, 2):
            raise ValueError(f"Expected shape ({self._num_points}, 2), got {points.shape}")
        super().add_sample(points.flatten())

    @property
    def value(self) -> np.ndarray:
        """Get smoothed points with shape (num_points, 2)."""
        return super().value.reshape(self._num_points, 2)


Smooth = Union[
    VectorSmooth,
    AngleSmooth,
    PointSmooth,
]