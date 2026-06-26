"""
=============================================================================
SINGLEANGLE — SINGLE ANGULAR VALUE FEATURE BASE (radians, wrapped to [-π, π))
=============================================================================

The single-element counterpart to the multi-element ``Angles`` feature, and the
angular sibling of ``NormalizedSingleValue`` / ``SingleValue``.

``from_value`` wraps its input to [-π, π) — the angular analogue of how
``NormalizedSingleValue.from_value`` clamps to [0, 1] — so producers can pass a
raw radian value (or an angular difference) and get a canonical angle back.

Use for: a single circular position/offset in radians (azimuth, playhead phase).
=============================================================================
"""
from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

from .BaseFeature import SYMMETRIC_PI_RANGE
from .SingleValue import SingleValue

if TYPE_CHECKING:
    from typing import Self


class SingleAngle(SingleValue):
    """Base for a single angular value in radians, wrapped to [-π, π)."""

    @classmethod
    def range(cls) -> tuple[float, float]:
        return SYMMETRIC_PI_RANGE

    @classmethod
    def from_value(cls, value: float, score: float = 1.0) -> Self:
        """Create from a single radian value, wrapped to [-π, π). NaN preserved.

        Uses ``arctan2(sin, cos)`` — the same self-contained wrap as ``Angles.subtract``.
        """
        wrapped = float(np.arctan2(np.sin(value), np.cos(value)))
        return cls(
            values=np.array([wrapped], dtype=np.float32),
            scores=np.array([score], dtype=np.float32),
        )
