from __future__ import annotations
from enum import IntEnum
from typing import TYPE_CHECKING

import numpy as np
from .NormalizedScalarFeature import NormalizedScalarFeature

if TYPE_CHECKING:
    from typing import Self


class SingleElement(IntEnum):
    """Generic single element enum."""
    VALUE = 0


class NormalizedSingleValue(NormalizedScalarFeature[SingleElement]):
    """Base for single normalized value features."""

    @classmethod
    def enum(cls) -> type[SingleElement]:
        return SingleElement

    @property
    def value(self) -> float:
        """Direct access to the single value."""
        return float(self._values[0])

    @property
    def score(self) -> float:
        """Direct access to the single score."""
        return float(self._scores[0])

    @classmethod
    def create_dummy(cls) -> Self:
        """Create a dummy instance with NaN value and zero score."""
        if cls._empty_instance is None:
            cls._empty_instance = cls(
                values=np.array([np.nan], dtype=np.float32),
                scores=np.array([0.0], dtype=np.float32)
            )
        return cls._empty_instance

    @classmethod
    def from_value(cls, value: float, score: float = 1.0) -> Self:
        """Create from a single value and score."""
        return cls(
            values=np.array([np.clip(value, 0.0, 1.0)], dtype=np.float32),
            scores=np.array([score], dtype=np.float32)
        )