# 2DO
# add lower and upper correlation
# apply weights to different joints

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Callable, Iterator, Optional

from modules.pose.features.PoseFeatureBase import PoseFeatureBase, FeatureStatistic
from modules.pose.features.PoseAngles import AngleJoint

@dataclass(frozen=True)
class PoseSimilarity(PoseFeatureBase[AngleJoint]):
    """Similarity scores between two poses for all joints."""

    pair_id: tuple[int, int] = field(default=(0, 0), kw_only=True)
    values: np.ndarray = field(default_factory=lambda: np.full(len(AngleJoint), np.nan, dtype=np.float32))
    scores: np.ndarray = field(default_factory=lambda: np.zeros(len(AngleJoint), dtype=np.float32))

    @property
    def joint_enum(self) -> type[AngleJoint]:
        """Return the AngleJoint enum class."""
        return AngleJoint

    def validate(self) -> None:
        """Validate pair_id and similarity values.

        Called automatically during construction.
        """
        # Validate pair_id ordering
        if self.pair_id[0] == self.pair_id[1]:
            raise ValueError(f"Cannot correlate pose with itself: {self.pair_id[0]}")
        if self.pair_id[0] > self.pair_id[1]:
            raise ValueError(f"pair_id must be ordered as (smaller, larger), got {self.pair_id}")

        # Validate similarity values are in [0, 1] range
        valid_values = self.values[~np.isnan(self.values)]
        if valid_values.size > 0:
            if np.any((valid_values < 0.0) | (valid_values > 1.0)):
                out_of_range = valid_values[(valid_values < 0.0) | (valid_values > 1.0)]
                raise ValueError(
                    f"Similarity values must be in range [0, 1]. "
                    f"Found values outside range: {out_of_range}"
                )

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
    pair_correlations: list[PoseSimilarity ]
    timestamp: pd.Timestamp = field(default_factory=pd.Timestamp.now)
    _pair_lookup: dict[tuple[int, int], PoseSimilarity ] = field(init=False, repr=False, compare=False, default_factory=dict)

    def __post_init__(self) -> None:
        """Build O(1) lookup index for pair access"""
        lookup: dict[tuple[int, int], PoseSimilarity ] = {pc.pair_id: pc for pc in self.pair_correlations}
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

    def __iter__(self) -> Iterator[PoseSimilarity ]:
        """Support 'for pair in batch' syntax"""
        return iter(self.pair_correlations)

    @property
    def is_empty(self) -> bool:
        """True if batch contains no pairs"""
        return len(self) == 0

    def get_pair(self, pair_id: tuple[int, int]) -> Optional[PoseSimilarity ]:
        """Get correlation for specific pair (O(1) lookup)."""
        pair_id = (min(pair_id), max(pair_id))
        return self._pair_lookup.get(pair_id)

    def get_top_pairs(self, n: int = 5, min_valid_joints: int = 1, metric: FeatureStatistic = FeatureStatistic.MEAN) -> list[PoseSimilarity ]:
        """Get top N most similar pairs, sorted by metric (descending)."""
        if self.is_empty or n <= 0:
            return []
        valid_pairs: list[PoseSimilarity ] = [pc for pc in self.pair_correlations if pc.valid_count >= min_valid_joints]
        return sorted(valid_pairs, key=lambda pc: pc.get_stat(metric), reverse=True)[:n]

PoseSimilarityBatchCallback = Callable[[PoseSimilarityBatch ], None]
