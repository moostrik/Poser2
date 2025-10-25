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
        correlations:   Dict mapping joint names to similarity scores [0.0, 1.0]
                        Can be empty if no overlapping joints detected.
        joint_weights:  Optional dict mapping joint names to weights (non-negative floats).
                        Can contain keys not in correlations (they're ignored).
                        Missing keys default to 1.0.
                        Common values: 0.0 (ignore), 0.5 (half weight), 1.0 (normal), 2.0 (double).
    """
    pair_id: tuple[int, int]
    correlations: dict[str, float]
    joint_weights: dict[str, float] | None = None

    @classmethod
    def from_ids(cls, id_1: int, id_2: int, correlations: dict[str, float], joint_weights: dict[str, float] | None = None ):
        """Create PairCorrelation from two tracklet IDs (auto-normalizes ordering).

        This is the recommended way to create PairCorrelation instances. The method
        automatically normalizes the pair_id to (smaller_id, larger_id) for consistent
        lookups and comparisons.

        Args:
            id_1:           First tracklet ID (order doesn't matter)
            id_2:           Second tracklet ID (order doesn't matter)
            correlations:   Dict mapping joint names to similarity scores [0.0, 1.0].
                            Can be empty if no overlapping joints detected.
            joint_weights:  Optional dict mapping joint names to weights (non-negative floats).
                            Can contain keys not in correlations (they're ignored).
                            Missing keys default to 1.0.
                            Common values: 0.0 (ignore), 0.5 (half), 1.0 (normal), 2.0 (double).
        """
        pair_id: tuple[int, int] = (id_1, id_2) if id_1 <= id_2 else (id_2, id_1)
        return cls(pair_id=pair_id, correlations=correlations, joint_weights=joint_weights)

    def __post_init__(self) -> None:
        """Validate pair, correlations and weights after initialization."""
        if self.id_1 == self.id_2:
            raise ValueError(f"Cannot correlate pose with itself: {self.id_1}")

        if self.id_1 > self.id_2:
            raise ValueError(f"pair_id must be ordered as (smaller_id, larger_id), got {self.pair_id}")

        # Validate correlation values are in [0, 1]
        for joint, value in self.correlations.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"Correlation for {joint} must be in [0, 1], got {value}")

        # Validate weights are non-negative (can be > 1.0)
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
        return (f"PairCorrelation(pair={self.id_1}-{self.id_2}, joints={self.joint_count}, mean={self.mean:.3f})")

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
        """Arithmetic mean - weighted if joint_weights provided"""
        if not self.correlations:
            return 0.0

        if self.joint_weights is None:
            return sum(self.correlations.values()) / self.joint_count

        weighted_sum: float = float(sum( self.correlations[joint] * self.joint_weights.get(joint, 1.0) for joint in self.correlations ))
        weight_total: float = sum(
            self.joint_weights.get(joint, 1.0)
            for joint in self.correlations
        )

        return weighted_sum / weight_total if weight_total > 0 else 0.0

    @cached_property
    def geometric_mean(self) -> float:
        """Geometric mean - weighted if joint_weights provided"""
        if not self.correlations:
            return 0.0

        if self.joint_weights is None:
            values: list[float] = [max(v, EPSILON) for v in self.correlations.values()]
            return float(gmean(values))

        # Weighted: exp(sum(w_i * log(x_i)) / sum(w_i))
        weights: list[float] = [self.joint_weights.get(joint, 1.0) for joint in self.correlations]
        weight_total: float = float(sum(weights))

        if weight_total == 0:
            return 0.0

        log_sum: float = float(sum(w * math.log(max(v, EPSILON)) for v, w, joint in zip(self.correlations.values(), weights, self.correlations.keys())))

        return math.exp(log_sum / weight_total)

    @cached_property
    def harmonic_mean(self) -> float:
        """Harmonic mean - weighted if joint_weights provided"""
        if not self.correlations:
            return 0.0

        if self.joint_weights is None:
            values: list[float] = [max(v, EPSILON) for v in self.correlations.values()]
            return float(hmean(values))

        # Weighted: sum(w_i) / sum(w_i / x_i)
        weights: list[float] = [self.joint_weights.get(joint, 1.0) for joint in self.correlations]
        weight_total: float = float(sum(weights))

        if weight_total == 0:
            return 0.0

        weighted_reciprocal_sum: float = float(sum(w / max(v, EPSILON) for v, w in zip(self.correlations.values(), weights)))

        return weight_total / weighted_reciprocal_sum if weighted_reciprocal_sum > 0 else 0.0

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
        """Standard deviation of joint similarities (always unweighted)"""
        if not self.correlations:
            return 0.0

        values: list[float] = list(self.correlations.values())
        unweighted_mean: float = sum(values) / len(values)
        variance: float = sum((v - unweighted_mean) ** 2 for v in values) / len(values)
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
