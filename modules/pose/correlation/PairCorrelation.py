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
                        Contains only joints available in both poses.
        joint_weights:  Optional dict mapping joint names to weights (non-negative floats).
                        Missing keys default to 1.0.

    Note:
        The correlations dict only includes joints that were successfully compared.
        Joints not available in both poses are simply not included in the dict.
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

        # Validate that correlations don't contain NaN
        for joint, value in self.correlations.items():
            if math.isnan(value):
                raise ValueError(f"Correlation for joint '{joint}' is NaN. Only include valid correlations.")

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
        """Total number of joints with valid correlations"""
        return len(self.correlations)

    @cached_property
    def valid_joint_count(self) -> int:
        """Number of joints with valid correlations (same as joint_count)"""
        return self.joint_count

    @cached_property
    def nan_joint_count(self) -> int:
        """Number of joints with NaN correlations (always 0)"""
        return 0

    @cached_property
    def matching_joint_count(self) -> int:
        """Number of joints with similarity > 0"""
        return sum(1 for v in self.correlations.values() if v > 0)

    @cached_property
    def is_empty(self) -> bool:
        """True if no correlations available"""
        return len(self.correlations) == 0

    @cached_property
    def mean(self) -> float:
        """Arithmetic mean of correlations. Weighted if joint_weights provided.

        Returns:
            Mean similarity [0.0, 1.0], or NaN if no correlations available.
        """
        if self.is_empty:
            return math.nan

        if self.joint_weights is None:
            return sum(self.correlations.values()) / len(self.correlations)

        # Weighted mean
        weighted_sum: float = float(sum(
            v * self.joint_weights.get(joint, 1.0) for joint, v in self.correlations.items()
        ))
        weight_total: float = float(sum(
            self.joint_weights.get(joint, 1.0) for joint in self.correlations.keys()
        ))
        return weighted_sum / weight_total if weight_total > 0 else math.nan

    @cached_property
    def geometric_mean(self) -> float:
        """Geometric mean of correlations. Weighted if joint_weights provided.

        Returns:
            Geometric mean [0.0, 1.0], or NaN if no correlations available.
        """
        if self.is_empty:
            return math.nan

        if self.joint_weights is None:
            values: list[float] = [max(v, EPSILON) for v in self.correlations.values()]
            return float(gmean(values))

        # Weighted geometric mean
        weights: list[float] = [self.joint_weights.get(joint, 1.0) for joint in self.correlations.keys()]
        values: list[float] = list(self.correlations.values())
        weight_total: float = float(sum(weights))

        if weight_total == 0:
            return 0.0

        log_sum: float = float(sum(w * math.log(max(v, EPSILON)) for v, w in zip(values, weights)))
        return math.exp(log_sum / weight_total)

    @cached_property
    def harmonic_mean(self) -> float:
        """Harmonic mean of correlations. Weighted if joint_weights provided.

        Returns:
            Harmonic mean [0.0, 1.0], or NaN if no correlations available.
        """
        if self.is_empty:
            return math.nan

        if self.joint_weights is None:
            values: list[float] = [max(v, EPSILON) for v in self.correlations.values()]
            return float(hmean(values))

        # Weighted harmonic mean
        weights: list[float] = [self.joint_weights.get(joint, 1.0) for joint in self.correlations.keys()]
        values: list[float] = list(self.correlations.values())
        weight_total: float = float(sum(weights))

        if weight_total == 0:
            return 0.0

        weighted_reciprocal_sum: float = float(sum(w / max(v, EPSILON) for v, w in zip(values, weights)))
        return weight_total / weighted_reciprocal_sum if weighted_reciprocal_sum > 0 else 0.0

    @cached_property
    def min_similarity(self) -> float:
        """Worst (minimum) joint similarity.

        Returns:
            Min similarity [0.0, 1.0], or NaN if no joints.
        """
        if self.is_empty:
            return math.nan
        return min(self.correlations.values())

    @cached_property
    def max_similarity(self) -> float:
        """Best (maximum) joint similarity.

        Returns:
            Max similarity [0.0, 1.0], or NaN if no joints.
        """
        if self.is_empty:
            return math.nan
        return max(self.correlations.values())

    @cached_property
    def std_deviation(self) -> float:
        """Standard deviation of joint similarities (always unweighted).

        Returns:
            Standard deviation, or NaN if < 2 joints.
        """
        if self.is_empty or len(self.correlations) < 2:
            return math.nan

        values: list[float] = list(self.correlations.values())
        mean_val: float = sum(values) / len(values)
        variance: float = sum((v - mean_val) ** 2 for v in values) / len(values)
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

    Note:
        All pair correlations are expected to have valid (non-NaN) metrics.
        Empty pairs (with no joints) may have NaN metrics, which are handled appropriately.
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
        mean_str = f"{self.mean:.3f}" if not math.isnan(self.mean) else "NaN"
        return f"PairCorrelationBatch(pairs={self.pair_count}, mean={mean_str}, timestamp={self.timestamp})"

    @cached_property
    def pair_count(self) -> int:
        """Total number of pairs in this batch"""
        return len(self.pair_correlations)

    @cached_property
    def is_empty(self) -> bool:
        """True if batch contains no pairs"""
        return self.pair_count == 0

    @cached_property
    def mean(self) -> float:
        """Average arithmetic mean across all non-empty pairs.

        Returns:
            Mean of all pair means [0.0, 1.0], or NaN if batch is empty or all pairs are empty.
        """
        if self.is_empty:
            return math.nan

        means: list[float] = [r.mean for r in self.pair_correlations if not r.is_empty]
        return sum(means) / len(means) if means else math.nan

    @cached_property
    def geometric_mean(self) -> float:
        """Geometric mean of all non-empty pair geometric means.

        Returns:
            Geometric mean [0.0, 1.0], or NaN if batch is empty or all pairs are empty.
        """
        if self.is_empty:
            return math.nan

        values: list[float] = [
            r.geometric_mean
            for r in self.pair_correlations
            if not r.is_empty
        ]
        return float(gmean(values)) if values else math.nan

    @cached_property
    def harmonic_mean(self) -> float:
        """Harmonic mean of all non-empty pair harmonic means.

        Returns:
            Harmonic mean [0.0, 1.0], or NaN if batch is empty or all pairs are empty.
        """
        if self.is_empty:
            return math.nan

        values: list[float] = [
            r.harmonic_mean
            for r in self.pair_correlations
            if not r.is_empty
        ]
        return float(hmean(values)) if values else math.nan

    def get_most_correlated_pair(self, metric: SimilarityMetric = SimilarityMetric.MEAN) -> Optional[PairCorrelation]:
        """Get the pair with highest similarity based on the specified metric.

        Args:
            metric: Which similarity metric to use

        Returns:
            PairCorrelation with highest metric value, or None if batch is empty.
            Empty pairs (with NaN metrics) are excluded.
        """
        if self.is_empty:
            return None

        non_empty_pairs = [pc for pc in self.pair_correlations if not pc.is_empty]
        if not non_empty_pairs:
            return None

        return max(non_empty_pairs, key=lambda r: r.get_metric_value(metric))

    def get_top_n_pairs(self, n: int = 5, metric: SimilarityMetric = SimilarityMetric.MEAN) -> list[PairCorrelation]:
        """Get top N most similar pairs.

        Args:
            n: Number of pairs to return
            metric: Which similarity metric to use

        Returns:
            List of up to N pairs sorted by metric (highest first).
            Empty pairs are excluded.
        """
        if self.is_empty:
            return []

        non_empty_pairs = [pc for pc in self.pair_correlations if not pc.is_empty]
        return sorted(
            non_empty_pairs,
            key=lambda r: r.get_metric_value(metric),
            reverse=True
        )[:n]

    def get_pair_metric(self, pair_id: tuple[int, int], metric: SimilarityMetric = SimilarityMetric.MEAN) -> float:
        """Get metric value for specific pair (O(1) lookup).

        Args:
            pair_id: Tuple of (id1, id2) in any order
            metric: Which similarity metric to use

        Returns:
            Metric value [0.0, 1.0], or NaN if pair not found or is empty.
        """
        pair_id = (min(pair_id), max(pair_id))
        pc: PairCorrelation | None = self._pair_lookup.get(pair_id)
        return pc.get_metric_value(metric) if pc else math.nan

    def get_pair(self, pair_id: tuple[int, int]) -> Optional[PairCorrelation]:
        """Get full correlation object for specific pair (O(1) lookup).

        Args:
            pair_id: Tuple of (id1, id2) in any order

        Returns:
            PairCorrelation object, or None if pair not found
        """
        pair_id = (min(pair_id), max(pair_id))
        return self._pair_lookup.get(pair_id)

PairCorrelationBatchCallback = Callable[[PairCorrelationBatch], None]
