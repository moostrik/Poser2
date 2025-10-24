# 2DO
# add lower and upper correlation
# apply weights to different joints

import math
import pandas as pd
from dataclasses import dataclass, field
from functools import cached_property
from typing import Callable, Iterator, Optional
from scipy.stats import hmean, gmean
from enum import Enum

EPSILON = 0.001


class SimilarityMetric(Enum):
    """Available similarity metrics for pose comparison"""
    MEAN = 'mean'
    GEOMETRIC_MEAN = 'geometric_mean'
    HARMONIC_MEAN = 'harmonic_mean'
    MIN = 'min_similarity'
    MAX = 'max_similarity'


@dataclass(frozen=True)
class PairCorrelation:
    """Similarity scores between two poses across multiple joints.

    Attributes:
        pair_id: Tuple of (smaller_id, larger_id) for consistent ordering
        correlations: Dict mapping joint names to similarity scores [0.0, 1.0]
    """
    pair_id: tuple[int, int]
    correlations: dict[str, float]

    def __len__(self) -> int:
        """Support len(pair) syntax"""
        return self.joint_count

    def __contains__(self, joint_name: str) -> bool:
        """Support 'joint_name in pair' syntax"""
        return joint_name in self.correlations

    def __iter__(self) -> Iterator[tuple[str, float]]:
        """Support 'for joint, value in pair' syntax"""
        return iter(self.correlations.items())

    @property
    def id_1(self) -> int:
        """First tracklet ID (always smaller)"""
        return self.pair_id[0]

    @property
    def id_2(self) -> int:
        """Second tracklet ID (always larger)"""
        return self.pair_id[1]

    @cached_property
    def joint_count(self) -> int:
        """Total number of joints compared"""
        return len(self.correlations)

    @cached_property
    def matching_joint_count(self) -> int:
        """Number of joints with similarity > 0"""
        return sum(1 for v in self.correlations.values() if v > 0)

    @cached_property
    def is_empty(self) -> bool:
        return self.joint_count == 0

    @cached_property
    def mean(self) -> float:
        """Arithmetic mean - general similarity across all joints"""
        return sum(self.correlations.values()) / self.joint_count if self.joint_count else 0.0

    @cached_property
    def geometric_mean(self) -> float:
        """Geometric mean - strict matching (penalizes mismatches)"""
        if not self.correlations:
            return 0.0
        values: list[float] = [max(v, EPSILON) for v in self.correlations.values()]
        return float(gmean(values)) if values else 0.0

    @cached_property
    def harmonic_mean(self) -> float:
        """Harmonic mean - anomaly detection (heavily penalizes outliers)"""
        if not self.correlations:
            return 0.0
        values: list[float] = [max(v, EPSILON) for v in self.correlations.values()]
        return float(hmean(values)) if values else 0.0

    @cached_property
    def min_similarity(self) -> float:
        """Worst (minimum) joint similarity"""
        return min(self.correlations.values()) if self.correlations else 0.0

    @cached_property
    def max_similarity(self) -> float:
        """Best (maximum) joint similarity"""
        return max(self.correlations.values()) if self.correlations else 0.0

    @cached_property
    def std_deviation(self) -> float:
        """Standard deviation of joint similarities"""
        if not self.correlations:
            return 0.0
        values = list(self.correlations.values())
        mean_val = self.mean
        variance = sum((v - mean_val) ** 2 for v in values) / len(values)
        return math.sqrt(variance)

    def get_metric_value(self, metric: SimilarityMetric) -> float:
        """Get the value for a specific similarity metric"""
        return getattr(self, metric.value)

    @classmethod
    def from_ids(cls, id_1: int, id_2: int, correlations: dict[str, float]):
        """Create PairCorrelation from two tracklet IDs and their joint correlations.

        Args:
            id_1: First tracklet ID
            id_2: Second tracklet ID
            correlations: Dict of joint name -> similarity score [0.0, 1.0]

        Raises:
            ValueError: If IDs are equal, correlations empty, or values out of range
        """
        if id_1 == id_2:
            raise ValueError(f"Cannot correlate pose with itself: {id_1}")
        if not correlations:
            raise ValueError("Correlations dictionary cannot be empty")

        # Validate all correlation values
        for joint, value in correlations.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"Correlation for {joint} must be in [0, 1], got {value}")

        # Create normalized pair_id
        pair_id: tuple[int, int] = (id_1, id_2) if id_1 <= id_2 else (id_2, id_1)
        return cls(pair_id=pair_id, correlations=correlations)


@dataclass(frozen=True)
class PairCorrelationBatch:
    """Collection of all pair correlations for a single frame.

    Attributes:
        pair_correlations: List of all pairwise correlations
        timestamp: When this batch was created
    """
    pair_correlations: list[PairCorrelation]
    timestamp: pd.Timestamp = field(default_factory=pd.Timestamp.now)
    _pair_lookup: dict[tuple[int, int], PairCorrelation] = field(init=False, repr=False, compare=False, default_factory=dict)

    def __post_init__(self) -> None:
        """Build O(1) lookup index for pair access"""
        lookup: dict[tuple[int, int], PairCorrelation] = {pc.pair_id: pc for pc in self.pair_correlations}
        object.__setattr__(self, '_pair_lookup', lookup)

    def __len__(self) -> int:
        """Support len(batch) syntax"""
        return self.pair_count

    def __contains__(self, pair_id: tuple[int, int]) -> bool:
        """Support 'pair_id in batch' syntax"""
        pair_id = (min(pair_id), max(pair_id))
        return pair_id in self._pair_lookup

    def __iter__(self) -> Iterator[PairCorrelation]:
        """Support 'for pair in batch' syntax"""
        return iter(self.pair_correlations)

    @cached_property
    def pair_count(self) -> int:
        """Total number of pairs in this batch"""
        return len(self.pair_correlations)

    @cached_property
    def is_empty(self) -> bool:
        return self.pair_count == 0

    @cached_property
    def mean(self) -> float:
        """Average arithmetic mean across all pairs, general similarity"""
        return sum(r.mean for r in self.pair_correlations) / self.pair_count if not self.is_empty else 0.0

    @cached_property
    def geometric_mean(self) -> float:
        """Geometric mean of all non-empty pair geometric means, strict similarity"""
        if self.is_empty:
            return 0.0
        values: list[float] = [r.geometric_mean for r in self.pair_correlations if r.geometric_mean > 0]
        return float(gmean(values)) if values else 0.0

    @cached_property
    def harmonic_mean(self) -> float:
        """Harmonic mean of all non-empty pair harmonic means, anomaly detection"""
        if self.is_empty:
            return 0.0
        values: list[float] = [r.harmonic_mean for r in self.pair_correlations if r.harmonic_mean > 0]
        return float(hmean(values)) if values else 0.0

    def get_most_correlated_pair(self,metric: SimilarityMetric = SimilarityMetric.MEAN) -> Optional[PairCorrelation]:
        """Get the pair with highest similarity based on the specified metric

        Args:
            metric: Which similarity metric to use
        """
        if self.is_empty:
            return None
        return max(self.pair_correlations, key=lambda r: r.get_metric_value(metric))

    def get_top_n_pairs(self,n: int = 5, metric: SimilarityMetric = SimilarityMetric.MEAN) -> list[PairCorrelation]:
        """Get top N most similar pairs.

        Args:
            n: Number of pairs to return
            metric: Which similarity metric to use
        """
        if self.is_empty:
            return []
        return sorted(
            self.pair_correlations,
            key=lambda r: r.get_metric_value(metric),
            reverse=True
        )[:n]

    def get_pair_metric(self, pair_id: tuple[int, int], metric: SimilarityMetric = SimilarityMetric.MEAN) -> float:
        """Get mean similarity for specific pair (O(1) lookup).

        Args:
            pair_id: Tuple of (id1, id2) in any order

        Returns:
            Mean similarity score, or 0.0 if pair not found
        """
        pair_id = (min(pair_id), max(pair_id))
        pc: PairCorrelation | None = self._pair_lookup.get(pair_id)
        return pc.get_metric_value(metric) if pc else 0.0

    def get_pair(self, pair_id: tuple[int, int]) -> Optional[PairCorrelation]:
        """Get full correlation object for specific pair (O(1) lookup).

        Args:
            pair_id: Tuple of (id1, id2) in any order

        Returns:
            PairCorrelation object, or None if pair not found
        """
        pair_id = (min(pair_id), max(pair_id))
        return self._pair_lookup.get(pair_id)


PoseCorrelationBatchCallback = Callable[[PairCorrelationBatch], None]
