"""
=============================================================================
SINGLEVALUE — UNBOUNDED SINGLE-FLOAT FEATURE BASE
=============================================================================

Like NormalizedSingleValue but values are NOT clamped to [0, 1].
Subclasses define their own range() (typically non-negative or unbounded).

Use for: accumulated motion time, elapsed age, or any single unbounded scalar.

Inherits all BaseScalarFeature infrastructure:
  • Immutable arrays, validity tracking, NaN semantics
  • .value / .score properties for direct access
  • from_value() factory, create_dummy() factory

Contrast with NormalizedSingleValue:
  • NormalizedSingleValue: range (0.0, 1.0), from_value clips to [0,1]
  • SingleValue: range defined by subclass, from_value does NOT clip
=============================================================================
"""
from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
from modules.pose.features.base.BaseScalarFeature import BaseScalarFeature
from modules.pose.features.base.NormalizedSingleValue import SingleElement

if TYPE_CHECKING:
    from typing_extensions import Self


class SingleValue(BaseScalarFeature[SingleElement]):
    """Base for single unbounded value features."""

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
        return cls(
            values=np.array([np.nan], dtype=np.float32),
            scores=np.array([0.0], dtype=np.float32),
        )

    @classmethod
    def from_value(cls, value: float, score: float = 1.0) -> Self:
        """Create from a single value and score. No clamping."""
        return cls(
            values=np.array([value], dtype=np.float32),
            scores=np.array([score], dtype=np.float32),
        )
