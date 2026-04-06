"""Leader score feature for temporal offset tracking between pose pairs.

Leader scores indicate which pose leads in synchronized movements:
- Negative value: this pose leads the other
- Positive value: other pose leads this one
- Zero: synchronized movement
"""

from enum import IntEnum
from typing import cast

import numpy as np

from modules.pose.features.base.BaseScalarFeature import BaseScalarFeature, FeatureEnum


# Module-level configuration (set once at app startup)
_PoseEnum: type[IntEnum] | None = None


class LeaderScore(BaseScalarFeature[FeatureEnum]):
    """Leader scores tracking temporal offset between pose pairs.

    Values in range [-1, 1]:
    - Negative: this pose leads the other
    - Positive: other pose leads this one
    - Zero: synchronized movement

    Scores represent confidence in the temporal alignment detection.
    """

    @classmethod
    def configure(cls, max_poses: int) -> None:
        """Configure with number of poses to track. Idempotent."""
        global _PoseEnum
        if _PoseEnum is None:
            _PoseEnum = cast(type[IntEnum], IntEnum("PoseIndex", {f"POSE_{i}": i for i in range(max_poses)}))

    def __init__(self, values: np.ndarray, scores: np.ndarray) -> None:
        if _PoseEnum is None:
            raise RuntimeError(
                "LeaderScore not configured. Call configure_features(max_poses) at app startup."
            )
        super().__init__(values, scores)

    @classmethod
    def enum(cls) -> type[IntEnum]:
        if _PoseEnum is None:
            raise RuntimeError(
                "LeaderScore not configured. Call configure_features(max_poses) at app startup."
            )
        return _PoseEnum

    @classmethod
    def range(cls) -> tuple[float, float]:
        """Returns (-1.0, 1.0) for leader scores."""
        return (-1.0, 1.0)

    @classmethod
    def create_dummy(cls) -> 'LeaderScore':
        """Create empty LeaderScore with all zero values and zero scores."""
        if cls._empty_instance is None:
            if _PoseEnum is None:
                raise RuntimeError(
                "LeaderScore not configured. Call configure_features(max_poses) at app startup."
                )
            values = np.zeros(len(_PoseEnum), dtype=np.float32)
            scores = np.zeros(len(_PoseEnum), dtype=np.float32)
            cls._empty_instance = cls(values, scores)
        return cls._empty_instance
