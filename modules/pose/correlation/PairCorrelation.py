# 2DO
# add lower and upper correlation
# apply weights to different joints

import math
import pandas as pd
from functools import cached_property
from dataclasses import dataclass, field
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
        pair_id:        Tuple of (smaller_id, larger_id) for consistent ordering.
        correlations:   Dict mapping joint names to similarity scores [0.0, 1.0].
                        Can contain NaN values for joints not available in both poses.
        joint_weights:  Optional dict mapping joint names to weights (non-negative floats).
                        Missing keys default to 1.0.

    Note:
        All statistical methods (mean, geometric_mean, etc.) automatically filter out
        NaN values, treating them as "joints not available for comparison" rather than
        "zero similarity". This ensures statistics reflect only comparable joints.
    """
    pair_id: tuple[int, int]
    correlations: dict[str, float]
    joint_weights: dict[str, float] | None = None

    @classmethod
    def from_ids(cls, id_1: int, id_2: int, correlations: dict[str, float], joint_weights: dict[str, float] | None = None):
        """Create PairCorrelation from two tracklet IDs (auto-normalizes ordering)."""
        pair_id: tuple[int, int] = (id_1, id_2) if id_1 <= id_2 else (id_2, id_1)
        return cls(pair_id=pair_id, correlations=correlations, joint_weights=joint_weights)

    def __post_init__(self) -> None:
        """Validate pair, correlations and weights after initialization."""
        if self.id_1 == self.id_2:
            raise ValueError(f"Cannot correlate pose with itself: {self.id_1}")
        if self.id_1 > self.id_2:
            raise ValueError(f"pair_id must be ordered as (smaller_id, larger_id), got {self.pair_id}")
        if self.joint_weights is not None:
            for joint, weight in self.joint_weights.items():
                if weight < 0:
                    raise ValueError(f"Weight for {joint} must be non-negative, got {weight}")

    def __len__(self) -> int:
        """Support len(pair) syntax"""
        return self.joint_count

    def __contains__(self, joint_name: str) -> bool:
        """Support 'joint_name in pair' syntax"""
        return joint_name in self.correlations

    def __iter__(self) -> Iterator[tuple[str, float]]:
        """Support 'for joint, value in pair' syntax"""
        return iter(self.correlations.items())

    def __repr__(self) -> str:
        """Readable string representation"""
        return (
            f"PairCorrelation(pair={self.id_1}-{self.id_2}, "
            f"joints={self.valid_joint_count}/{self.joint_count}, "
            f"mean={self.mean:.3f})"
        )

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
        """Total number of joints in correlations dict (includes NaN)"""
        return len(self.correlations)

    @cached_property
    def valid_joint_count(self) -> int:
        """Number of joints with valid (non-NaN) correlations"""
        return sum(1 for v in self.correlations.values() if not math.isnan(v))

    @cached_property
    def nan_joint_count(self) -> int:
        """Number of joints with NaN correlations (not comparable)"""
        return self.joint_count - self.valid_joint_count

    @cached_property
    def matching_joint_count(self) -> int:
        """Number of valid joints with similarity > 0"""
        return sum(1 for v in self.correlations.values() if not math.isnan(v) and v > 0)

    @cached_property
    def is_empty(self) -> bool:
        """True if no valid correlations available"""
        return self.valid_joint_count == 0

    @cached_property
    def mean(self) -> float:
        """Arithmetic mean of valid correlations. Weighted if joint_weights provided. Ignores NaN.

        Returns:
            Mean similarity [0.0, 1.0], or NaN if no valid correlations available.
        """
        if self.is_empty:
            return math.nan

        if self.joint_weights is None:
            valid_values: list[float] = [v for v in self.correlations.values() if not math.isnan(v)]
            return sum(valid_values) / len(valid_values)

        # Weighted mean
        valid_items: list[tuple[str, float]] = [
            (joint, v) for joint, v in self.correlations.items() if not math.isnan(v)
        ]
        weighted_sum: float = float(sum(
            v * self.joint_weights.get(joint, 1.0) for joint, v in valid_items
        ))
        weight_total: float = float(sum(
            self.joint_weights.get(joint, 1.0) for joint, _ in valid_items
        ))
        return weighted_sum / weight_total if weight_total > 0 else math.nan

    @cached_property
    def geometric_mean(self) -> float:
        """Geometric mean of valid correlations. Weighted if joint_weights provided. Ignores NaN.

        Returns:
            Geometric mean [0.0, 1.0], or NaN if no valid correlations available.
        """
        if self.is_empty:
            return math.nan

        if self.joint_weights is None:
            valid_values: list[float] = [
                max(v, EPSILON) for v in self.correlations.values() if not math.isnan(v)
            ]
            return float(gmean(valid_values))

        # Weighted geometric mean
        valid_items: list[tuple[str, float]] = [
            (joint, v) for joint, v in self.correlations.items() if not math.isnan(v)
        ]
        weights: list[float] = [self.joint_weights.get(joint, 1.0) for joint, _ in valid_items]
        values: list[float] = [v for _, v in valid_items]
        weight_total: float = float(sum(weights))

        if weight_total == 0:
            return 0.0

        log_sum: float = float(sum(w * math.log(max(v, EPSILON)) for v, w in zip(values, weights)))
        return math.exp(log_sum / weight_total)

    @cached_property
    def harmonic_mean(self) -> float:
        """Harmonic mean of valid correlations. Weighted if joint_weights provided. Ignores NaN.

        Returns:
            Harmonic mean [0.0, 1.0], or NaN if no valid correlations available.
        """
        if self.is_empty:
            return math.nan

        if self.joint_weights is None:
            valid_values: list[float] = [
                max(v, EPSILON) for v in self.correlations.values() if not math.isnan(v)
            ]
            return float(hmean(valid_values))

        # Weighted harmonic mean
        valid_items: list[tuple[str, float]] = [
            (joint, v) for joint, v in self.correlations.items() if not math.isnan(v)
        ]
        weights: list[float] = [self.joint_weights.get(joint, 1.0) for joint, _ in valid_items]
        values: list[float] = [v for _, v in valid_items]
        weight_total: float = float(sum(weights))

        if weight_total == 0:
            return 0.0

        weighted_reciprocal_sum: float = float(sum(w / max(v, EPSILON) for v, w in zip(values, weights)))
        return weight_total / weighted_reciprocal_sum if weighted_reciprocal_sum > 0 else 0.0

    @cached_property
    def min_similarity(self) -> float:
        """Worst (minimum) valid joint similarity. Ignores NaN.

        Returns:
            Min similarity [0.0, 1.0], or NaN if no valid joints.
        """
        if self.is_empty:
            return math.nan
        valid_values: list[float] = [v for v in self.correlations.values() if not math.isnan(v)]
        return min(valid_values)

    @cached_property
    def max_similarity(self) -> float:
        """Best (maximum) valid joint similarity. Ignores NaN.

        Returns:
            Max similarity [0.0, 1.0], or NaN if no valid joints.
        """
        if self.is_empty:
            return math.nan
        valid_values: list[float] = [v for v in self.correlations.values() if not math.isnan(v)]
        return max(valid_values)

    @cached_property
    def std_deviation(self) -> float:
        """Standard deviation of valid joint similarities (always unweighted). Ignores NaN.

        Returns:
            Standard deviation, or NaN if < 2 valid joints.
        """
        if self.is_empty:
            return math.nan

        valid_values: list[float] = [v for v in self.correlations.values() if not math.isnan(v)]

        if len(valid_values) < 2:
            return math.nan

        unweighted_mean: float = sum(valid_values) / len(valid_values)
        variance: float = sum((v - unweighted_mean) ** 2 for v in valid_values) / len(valid_values)
        return math.sqrt(variance)

    def get_metric_value(self, metric: SimilarityMetric) -> float:
        """Get the value for a specific similarity metric"""
        return getattr(self, metric.value)


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

    def __repr__(self) -> str:
        """Readable string representation"""
        return (f"PairCorrelationBatch(pairs={self.pair_count}, mean={self.mean:.3f}, timestamp={self.timestamp})")

    @cached_property
    def pair_count(self) -> int:
        """Total number of pairs in this batch"""
        return len(self.pair_correlations)

    @cached_property
    def is_empty(self) -> bool:
        return self.pair_count == 0

    @cached_property
    def mean(self) -> float:
        """Average arithmetic mean across all pairs. Returns NaN if batch is empty."""
        if self.is_empty:
            return math.nan
        valid_means: list[float] = [r.mean for r in self.pair_correlations if not math.isnan(r.mean)]
        if not valid_means:
            return math.nan
        return sum(valid_means) / len(valid_means)

    @cached_property
    def geometric_mean(self) -> float:
        """Geometric mean of all non-empty pair geometric means. Returns NaN if batch is empty."""
        if self.is_empty:
            return math.nan
        values: list[float] = [
            r.geometric_mean
            for r in self.pair_correlations
            if not math.isnan(r.geometric_mean) and r.geometric_mean > 0
        ]
        if not values:
            return math.nan
        return float(gmean(values))

    @cached_property
    def harmonic_mean(self) -> float:
        """Harmonic mean of all non-empty pair harmonic means. Returns NaN if batch is empty."""
        if self.is_empty:
            return math.nan
        values: list[float] = [
            r.harmonic_mean
            for r in self.pair_correlations
            if not math.isnan(r.harmonic_mean) and r.harmonic_mean > 0
        ]
        if not values:
            return math.nan
        return float(hmean(values))

    def get_most_correlated_pair(self, metric: SimilarityMetric = SimilarityMetric.MEAN) -> Optional[PairCorrelation]:
        """Get the pair with highest similarity based on the specified metric

        Args:
            metric: Which similarity metric to use
        """
        if self.is_empty:
            return None
        return max(self.pair_correlations, key=lambda r: r.get_metric_value(metric))

    def get_top_n_pairs(self, n: int = 5, metric: SimilarityMetric = SimilarityMetric.MEAN) -> list[PairCorrelation]:
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
