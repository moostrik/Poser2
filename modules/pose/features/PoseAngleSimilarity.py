# Standard library imports
from dataclasses import dataclass, field
import time
from typing import Callable, Iterator, Optional

# Third-party imports
import numpy as np

# Pose imports
from modules.pose.features.PoseAngles import AngleJoint
from modules.pose.features.PoseAngleFeatureBase import PoseAngleFeatureBase, FeatureStatistic

@dataclass(frozen=True)
class PoseAngleSimilarityData(PoseAngleFeatureBase[AngleJoint]):
    """Similarity scores between two poses for all joints."""

    pair_id: tuple[int, int]

    # ========== INSTANCE-LEVEL METHODS ==========

    def __post_init__(self) -> None:
        """Validate data integrity."""

        if self.pair_id[0] == self.pair_id[1]:
            raise ValueError(f"Cannot correlate pose with itself: {self.pair_id[0]}")
        if self.pair_id[0] > self.pair_id[1]:
            raise ValueError(f"pair_id must be ordered as (smaller, larger), got {self.pair_id}")

        super().__post_init__()

    # ========== CLASS-LEVEL PROPERTIES ==========

    @classmethod
    def joint_enum(cls) -> type[AngleJoint]:
        """Return the AngleJoint enum class."""
        return AngleJoint

    @classmethod
    def default_range(cls) -> tuple[float, float]:
        """Return the default range for angle joints."""
        return (-np.pi, np.pi)

    # ========== PROPERTIES ==========
    @property
    def id_1(self) -> int:
        """First tracklet ID (always smaller)."""
        return self.pair_id[0]

    @property
    def id_2(self) -> int:
        """Second tracklet ID (always larger)."""
        return self.pair_id[1]


@dataclass(frozen=True)
class PoseSimilarityBatch:
    """Collection of all pair correlations for a single frame.

    Simple container with O(1) lookup and iteration support.
    """
    pair_correlations: list[PoseAngleSimilarityData ]
    timestamp: float = field(default_factory=time.time)
    _pair_lookup: dict[tuple[int, int], PoseAngleSimilarityData ] = field(init=False, repr=False, compare=False, default_factory=dict)

    def __post_init__(self) -> None:
        """Build O(1) lookup index for pair access"""
        lookup: dict[tuple[int, int], PoseAngleSimilarityData ] = {pc.pair_id: pc for pc in self.pair_correlations}
        object.__setattr__(self, '_pair_lookup', lookup)

    def __repr__(self) -> str:
        return f"PairCorrelationBatch({len(self)} pairs, {self.timestamp})"

    def __len__(self) -> int:
        """Number of pairs in batch."""
        return len(self.pair_correlations)

    def __contains__(self, pair_id: tuple[int, int]) -> bool:
        """Support 'pair_id in batch' syntax"""
        pair_id = (min(pair_id), max(pair_id))
        return pair_id in self._pair_lookup

    def __iter__(self) -> Iterator[PoseAngleSimilarityData ]:
        """Support 'for pair in batch' syntax"""
        return iter(self.pair_correlations)

    @property
    def is_empty(self) -> bool:
        """True if batch contains no pairs"""
        return len(self) == 0

    def get_pair(self, pair_id: tuple[int, int]) -> Optional[PoseAngleSimilarityData ]:
        """Get correlation for specific pair (O(1) lookup)."""
        pair_id = (min(pair_id), max(pair_id))
        return self._pair_lookup.get(pair_id)

    def get_top_pairs(self, n: int = 5, min_valid_joints: int = 1, metric: FeatureStatistic = FeatureStatistic.MEAN) -> list[PoseAngleSimilarityData ]:
        """Get top N most similar pairs, sorted by metric (descending)."""
        if self.is_empty or n <= 0:
            return []
        valid_pairs: list[PoseAngleSimilarityData ] = [pc for pc in self.pair_correlations if pc.valid_count >= min_valid_joints]
        return sorted(valid_pairs, key=lambda pc: pc.get_stat(metric), reverse=True)[:n]

PoseSimilarityBatchCallback = Callable[[PoseSimilarityBatch ], None]

