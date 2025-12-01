from enum import IntEnum
import numpy as np
from modules.pose.features.base.NormalizedScalarFeature import NormalizedScalarFeature


class SingleElement(IntEnum):
    """Generic single element enum."""
    VALUE = 0


class SingleValueFeature(NormalizedScalarFeature[SingleElement]):
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
    def create_dummy(cls) -> 'SingleValueFeature':
        """Create from a single value and score."""
        return cls(
            values=np.array([np.nan], dtype=np.float32),
            scores=np.array([0.0], dtype=np.float32)
        )