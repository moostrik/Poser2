"""AngleSymmetry — how unequal the left and right sides of the body are, signed, per pair.

One value per element, left minus right, in [-1, 1]: 0 is a mirror-symmetric pose, the sign
says which side leads (positive: the left angle is the larger), the magnitude how far apart they
are. The four joints are the wrapped angle difference over π; ``arms`` is the mean of the
shoulder and elbow elements, ``legs`` the mean of the hip and knee differences each normalised
by its own full-deviation angle (as ``LegDeviation`` weights them), clipped to the range.
Populated by ``AngleSymExtractor`` from ``Angles`` (already mirrored, so a symmetric pose gives
left == right). An element is NaN with score 0.0 when either side is missing.
"""

from __future__ import annotations

from enum import IntEnum

from .base import BaseScalarFeature

SYMMETRY_RANGE: tuple[float, float] = (-1.0, 1.0)


class SymmetryElement(IntEnum):
    """The pairs compared: the four joints, and the arm and the leg as a whole per side."""
    shoulder = 0
    elbow    = 1
    hip      = 2
    knee     = 3
    arms     = 4     # shoulder and elbow together
    legs     = 5     # hip and knee together, weighted as the leg deviation


class AngleSymmetry(BaseScalarFeature[SymmetryElement]):
    """Signed left-minus-right per pair, in [-1, 1]; see the module docstring."""

    @classmethod
    def enum(cls) -> type[IntEnum]:
        return SymmetryElement

    @classmethod
    def range(cls) -> tuple[float, float]:
        return SYMMETRY_RANGE
