"""
=============================================================================
MOTIONGATE FEATURE API REFERENCE
=============================================================================

Pairwise motion gate feature for synchronized movement detection.

Use for: detecting when two poses are both moving simultaneously.

The MotionGate value between pose A and pose B is computed as:
    gate[A, B] = motion_A * motion_B

Where motion is the max aggregated AngleMotion value for each pose.

Values:
  • 0.0 = at least one pose is stationary (or missing)
  • 1.0 = both poses have maximum motion simultaneously
  • Intermediate values indicate partial synchronized movement

Storage:
--------
Like Similarity, values are indexed by the OTHER pose's track_id.
For pose A with track_id=2:
  • motion_gate.values[0] = gate between pose 2 and pose with track_id 0
  • motion_gate.values[1] = gate between pose 2 and pose with track_id 1

Missing poses have motion treated as 0.0, so gate becomes 0.0.

=============================================================================
"""

from enum import IntEnum
from typing import cast

import numpy as np

from modules.pose.features.base.NormalizedScalarFeature import NormalizedScalarFeature, AggregationMethod


# Module-level configuration (set once at app startup)
_PoseEnum: type[IntEnum] | None = None


def configure_motion_gate(max_poses: int) -> None:
    """Configure MotionGate feature with number of poses to track.

    Must be called once at application initialization before creating any Frame instances.

    Args:
        max_poses: Maximum number of poses to compare motion gates for
    """
    global _PoseEnum
    if _PoseEnum is None:
        _PoseEnum = cast(type[IntEnum], IntEnum("PoseIndex", {f"POSE_{i}": i for i in range(max_poses)}))


class MotionGate(NormalizedScalarFeature):
    """Motion gate scores between current pose and other tracked poses.

    Represents synchronized movement: high values when both poses move together,
    low/zero values when either pose is stationary or missing.
    """

    def __init__(self, values: np.ndarray, scores: np.ndarray) -> None:
        if _PoseEnum is None:
            raise RuntimeError(
                "MotionGate not configured. Call configure_motion_gate(max_poses) at app startup."
            )
        super().__init__(values, scores)

    @classmethod
    def enum(cls) -> type[IntEnum]:
        if _PoseEnum is None:
            raise RuntimeError(
                "MotionGate not configured. Call configure_motion_gate(max_poses) at app startup."
            )
        return _PoseEnum

    @classmethod
    def create_dummy(cls) -> 'MotionGate':
        """Create a dummy MotionGate with all zeros (no motion gate computed)."""
        if _PoseEnum is None:
            # Return minimal dummy if not configured yet
            return cls.__new__(cls)
        n = len(_PoseEnum)
        return cls(
            values=np.zeros(n, dtype=np.float32),
            scores=np.zeros(n, dtype=np.float32)
        )

    def overall_gate(self) -> float:
        """Compute overall motion gate using max (highest synchronized movement)."""
        return self.aggregate(method=AggregationMethod.MAX, min_confidence=0.0)
