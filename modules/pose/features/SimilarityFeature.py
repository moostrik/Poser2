from enum import IntEnum
import time
from dataclasses import dataclass, field
from typing import Iterator, Optional, Callable

import numpy as np

from modules.pose.features.base.BaseFeature import NORMALIZED_RANGE
from modules.pose.features.base.NormalizedScalarFeature import NormalizedScalarFeature, AggregationMethod
from modules.pose.features.AngleFeature import AngleLandmark


# Reuse AngleLandmark enum for similarity (same joints as angles)
# No need to define a new enum since similarity is measured per angle landmark

# Constants
SIMILARITY_RANGE: tuple[float, float] = NORMALIZED_RANGE


class SimilarityFeature(NormalizedScalarFeature[AngleLandmark]):
    """Similarity scores between two poses for all angle landmarks (range [0, 1]).

    Measures how similar corresponding joint angles are between two poses.
    Values are similarity scores [0, 1] per joint:
    - 1.0 = perfect similarity (angles identical)
    - 0.0 = maximum dissimilarity (angles maximally different, e.g., π radians apart)

    Each similarity score has a confidence based on the minimum confidence
    of the two angles being compared.

    The pair_id tracks which two poses are being compared (always ordered as smaller, larger).

    Inherits statistical aggregation methods from NormalizedScalarFeature:
    - mean(), geometric_mean(), harmonic_mean()
    - aggregate() with multiple methods
    - All methods support confidence filtering

    Common use cases:
    - Pose matching: Compare if two people are doing the same pose
    - Pose tracking: Verify pose consistency across frames
    - Quality assessment: Check if detected pose matches reference pose
    """

    def __init__(self,
                 values: np.ndarray,
                 scores: np.ndarray,
                 pair_id: tuple[int, int]):
        """Initialize SimilarityFeature with pair tracking.

        Args:
            values: Similarity scores [0, 1] for each joint
            scores: Confidence scores for each similarity measurement
            pair_id: Tuple of (pose_id_1, pose_id_2) being compared

        Raises:
            ValueError: If pair_id is invalid (same ID or not ordered)
        """
        # Validate pair_id -> this should assert not raise an error
        if pair_id[0] == pair_id[1]:
            raise ValueError(f"Cannot compare pose with itself: {pair_id[0]}")
        if pair_id[0] > pair_id[1]:
            raise ValueError(f"pair_id must be ordered as (smaller, larger), got {pair_id}")

        super().__init__(values, scores)
        self._pair_id = pair_id

    # ========== ABSTRACT METHOD IMPLEMENTATIONS ==========

    @classmethod
    def feature_enum(cls) -> type[AngleLandmark]:
        """Returns AngleLandmark enum."""
        return AngleLandmark

    # ========== PAIR TRACKING PROPERTIES ==========

    @property
    def pair_id(self) -> tuple[int, int]:
        """ID pair of the two poses being compared (always ordered)."""
        return self._pair_id

    @property
    def id_1(self) -> int:
        """First pose ID (always smaller)."""
        return self._pair_id[0]

    @property
    def id_2(self) -> int:
        """Second pose ID (always larger)."""
        return self._pair_id[1]

    # ========== SIMILARITY-SPECIFIC CONVENIENCE METHODS ==========

    def similarity(self, method: AggregationMethod = AggregationMethod.MEAN,
                          min_confidence: float = 0.0) -> float:
        """Compute overall similarity score using specified aggregation method.

        Args:
            method: Statistical aggregation method (default: MEAN)
            min_confidence: Minimum confidence to include joint (default: 0.0)

        Returns:
            Overall similarity score [0, 1], or NaN if no joints meet criteria

        Examples:
            >>> # Mean similarity (default, balanced)
            >>> overall = similarity.similarity()
            >>>
            >>> # Geometric mean (penalizes dissimilar joints more)
            >>> overall = similarity.similarity(AggregationMethod.GEOMETRIC_MEAN)
            >>>
            >>> # Harmonic mean (very strict - heavily penalizes any dissimilarity)
            >>> overall = similarity.similarity(AggregationMethod.HARMONIC_MEAN)
            >>>
            >>> # Only trust high-confidence measurements
            >>> overall = similarity.similarity(
            ...     AggregationMethod.MEAN,
            ...     min_confidence=0.7
            ... )
            >>>
            >>> # For strict pose matching (all joints must match):
            >>> overall = similarity.similarity(
            ...     AggregationMethod.HARMONIC_MEAN,
            ...     min_confidence=0.6
            ... )
        """
        return self.aggregate(method, min_confidence)

    def matches_pose(self, threshold: float = 0.8,
                     method: AggregationMethod = AggregationMethod.GEOMETRIC_MEAN,
                     min_confidence: float = 0.0) -> bool:
        """Check if poses match based on overall similarity threshold.

        Args:
            threshold: Minimum similarity to consider poses matching (default: 0.8)
            method: Aggregation method (default: GEOMETRIC_MEAN for balanced strictness)
            min_confidence: Minimum confidence to include joint (default: 0.0)

        Returns:
            True if overall similarity >= threshold, False otherwise

        Examples:
            >>> # Default: 80% similarity with geometric mean
            >>> if similarity.matches_pose():
            ...     print("Poses match!")
            >>>
            >>> # Strict matching: 90% with harmonic mean
            >>> if similarity.matches_pose(0.9, AggregationMethod.HARMONIC_MEAN):
            ...     print("Poses match very closely!")
            >>>
            >>> # Lenient matching: 70% with mean
            >>> if similarity.matches_pose(0.7, AggregationMethod.MEAN):
            ...     print("Poses are somewhat similar")
        """
        overall = self.similarity(method, min_confidence)
        return not np.isnan(overall) and overall >= threshold


    # ========== REPRESENTATION ==========

    def __repr__(self) -> str:
        """String representation showing pair IDs and overall similarity."""
        overall = self.similarity()
        valid = self.valid_count
        total = len(self)
        return (f"SimilarityFeature(pair={self.pair_id}, "
                f"similarity={overall:.3f}, valid={valid}/{total})")

# ========== BATCH COLLECTION ==========

@dataclass(frozen=True)
class SimilarityBatch:
    """Collection of all similarity features for multiple pose pairs in a frame.

    Simple container with O(1) lookup and iteration support.
    Stores similarity comparisons between all pose pairs detected in a frame.

    Attributes:
        similarities: List of SimilarityFeature objects (one per pose pair)
        timestamp: When this batch was created (default: current time)

    Examples:
        >>> # Create batch
        >>> batch = SimilarityFeatureBatch([sim1, sim2, sim3])
        >>>
        >>> # Check size
        >>> print(f"Comparing {len(batch)} pose pairs")
        >>>
        >>> # Lookup specific pair
        >>> sim = batch.get_pair((5, 8))
        >>> if sim:
        ...     print(f"Similarity: {sim.similarity():.2f}")
        >>>
        >>> # Iterate all pairs
        >>> for similarity in batch:
        ...     print(f"{similarity.pair_id}: {similarity.similarity():.2f}")
        >>>
        >>> # Get top matches
        >>> top_5 = batch.get_top_pairs(n=5)
        >>> for sim in top_5:
        ...     print(f"Pair {sim.pair_id}: {sim.similarity():.2f}")
    """
    similarities: list[SimilarityFeature]
    timestamp: float = field(default_factory=time.time)
    _pair_lookup: dict[tuple[int, int], SimilarityFeature] = field(
        init=False, repr=False, compare=False, default_factory=dict
    )

    def __post_init__(self) -> None:
        """Build O(1) lookup index for pair access."""
        lookup: dict[tuple[int, int], SimilarityFeature] = {
            sim.pair_id: sim for sim in self.similarities
        }
        object.__setattr__(self, '_pair_lookup', lookup)

    def __repr__(self) -> str:
        return f"SimilarityFeatureBatch({len(self)} pairs, timestamp={self.timestamp:.3f})"

    def __len__(self) -> int:
        """Number of pairs in batch."""
        return len(self.similarities)

    def __contains__(self, pair_id: tuple[int, int]) -> bool:
        """Support 'pair_id in batch' syntax.

        Args:
            pair_id: Tuple of (id_1, id_2), will be normalized to (min, max)

        Returns:
            True if pair exists in batch

        Examples:
            >>> if (5, 8) in batch:
            ...     print("Found pair (5, 8)")
        """
        pair_id = (min(pair_id), max(pair_id))
        return pair_id in self._pair_lookup

    def __iter__(self) -> Iterator[SimilarityFeature]:
        """Support 'for similarity in batch' syntax."""
        return iter(self.similarities)

    @property
    def is_empty(self) -> bool:
        """True if batch contains no pairs."""
        return len(self) == 0

    def get_pair(self, pair_id: tuple[int, int]) -> Optional[SimilarityFeature]:
        """Get similarity for specific pair (O(1) lookup).

        Args:
            pair_id: Tuple of (id_1, id_2), will be normalized to (min, max)

        Returns:
            SimilarityFeature if pair exists, None otherwise

        Examples:
            >>> sim = batch.get_pair((5, 8))
            >>> if sim:
            ...     print(f"Similarity: {sim.similarity():.2f}")
        """
        pair_id = (min(pair_id), max(pair_id))
        return self._pair_lookup.get(pair_id)

    def get_top_pairs(
        self,
        n: int = 5,
        min_valid_joints: int = 1,
        method: AggregationMethod = AggregationMethod.MEAN,
        min_confidence: float = 0.0
    ) -> list[SimilarityFeature]:
        """Get top N most similar pairs, sorted by similarity (descending).

        Args:
            n: Number of top pairs to return (default: 5)
            min_valid_joints: Minimum valid joints required (default: 1)
            method: Aggregation method for sorting (default: MEAN)
            min_confidence: Minimum confidence for aggregation (default: 0.0)

        Returns:
            List of top N most similar pairs, sorted best-first

        Examples:
            >>> # Get top 5 most similar pairs
            >>> top = batch.get_top_pairs(n=5)
            >>> for sim in top:
            ...     print(f"{sim.pair_id}: {sim.similarity():.2f}")
            >>>
            >>> # Get top pairs with strict criteria
            >>> top = batch.get_top_pairs(
            ...     n=3,
            ...     min_valid_joints=10,
            ...     method=AggregationMethod.GEOMETRIC_MEAN,
            ...     min_confidence=0.7
            ... )
        """
        if self.is_empty or n <= 0:
            return []

        # Filter by minimum valid joints
        valid_pairs = [
            sim for sim in self.similarities
            if sim.valid_count >= min_valid_joints
        ]

        # Compute scores and filter out NaN
        scored_pairs: list[tuple[SimilarityFeature, float]] = []
        for sim in valid_pairs:
            score = sim.aggregate(method, min_confidence)
            if not np.isnan(score):
                scored_pairs.append((sim, score))

        # Sort by score (descending)
        scored_pairs.sort(key=lambda x: x[1], reverse=True)

        return [sim for sim, _ in scored_pairs[:n]]


# Type alias for batch callbacks
SimilarityBatchCallback = Callable[[SimilarityBatch], None]



"""
=============================================================================
SIMILARITYFEATURE QUICK API REFERENCE
=============================================================================

Design Philosophy (from BaseFeature):
-------------------------------------
Raw Access (numpy-native):
  • feature.values      → Full array, shape (n_joints,) for similarity scores
  • feature.scores      → Full confidence scores (n_joints,)
  • feature[joint]      → Single value (float, [0, 1])
  Use for: Numpy operations, batch processing, performance

Python-Friendly Access:
  • feature.get(joint, fill)    → Python float with NaN handling
  • feature.get_score(joint)    → Python float
  • feature.get_scores(joints)  → Python list
  Use for: Logic, conditionals, unpacking, defaults

Pair Tracking:
--------------
  • feature.pair_id      → (id_1, id_2) tuple identifying the two poses
  • feature.id_1         → First pose ID (always smaller)
  • feature.id_2         → Second pose ID (always larger)

Inherited from BaseScalarFeature (single value per joint):
----------------------------------------------------------
Properties:
  • values: np.ndarray                             All similarity scores [0, 1]
  • scores: np.ndarray                             All confidence scores
  • valid_mask: np.ndarray                         Boolean validity mask
  • valid_count: int                               Number of valid similarity scores
  • len(feature): int                              Total number of joints (17)

Single Value Access:
  • feature[joint] -> float                        Get similarity score [0, 1]
  • feature.get(joint, fill=0.0) -> float          Get score with NaN fill
  • feature.get_value(joint, fill) -> float        Alias for get()
  • feature.get_score(joint) -> float              Get confidence score
  • feature.get_valid(joint) -> bool               Check if score is valid

Batch Operations:
  • feature.get_values(joints, fill) -> list[float]  Get multiple scores
  • feature.get_scores(joints) -> list[float]        Get multiple confidences
  • feature.are_valid(joints) -> bool                Check if ALL valid

Factory Methods:
  • feature.create_empty() -> feature             All NaN scores
  • feature.from_values(values, scores)           Create with validation
  • feature.create_validated(values, scores)      Create with strict checks

Inherited from NormalizedScalarFeature (statistical aggregation):
------------------------------------------------------------------
Statistical Methods:
  • feature.mean(min_confidence=0.0) -> float
      Confidence-weighted arithmetic mean
      Use for: General purpose, balanced averaging

  • feature.geometric_mean(min_confidence=0.0) -> float
      Confidence-weighted geometric mean (penalizes low values)
      Use for: When most joints should match

  • feature.harmonic_mean(min_confidence=0.0) -> float
      Confidence-weighted harmonic mean (heavily penalizes low values)
      Use for: When ALL joints must match (strict)

  • feature.aggregate(method, min_confidence=0.0) -> float
      General aggregation with method selection

  • feature.min_value(min_confidence=0.0) -> float
      Minimum similarity score (worst matching joint)

  • feature.max_value(min_confidence=0.0) -> float
      Maximum similarity score (best matching joint)

  • feature.median(min_confidence=0.0) -> float
      Median similarity score

  • feature.std(min_confidence=0.0) -> float
      Standard deviation of similarity scores

SimilarityFeature-Specific Methods:
-----------------------------------------
  • feature.similarity(method=MEAN, min_confidence=0.0) -> float
      Convenience wrapper for aggregate() with similarity-specific naming
      Recommended method names for similarity:
      - AggregationMethod.MEAN: Balanced overall similarity
      - AggregationMethod.GEOMETRIC_MEAN: Penalize dissimilar joints
      - AggregationMethod.HARMONIC_MEAN: Strict matching requirement

  • feature.matches_pose(threshold=0.8, method=GEOMETRIC_MEAN, min_confidence=0.0) -> bool
      Quick check if poses match above threshold
      Returns True if similarity >= threshold

Common Usage Patterns:
----------------------
# Create similarity feature for pose pair (5, 8):
similarity = SimilarityFeature(values, scores, pair_id=(5, 8))

# Check which poses are being compared:
print(f"Comparing poses {similarity.id_1} and {similarity.id_2}")

# Individual joint similarity:
shoulder_sim = similarity[AngleLandmark.left_shoulder]
if shoulder_sim > 0.9:
    print("Left shoulder matches well!")

# Overall similarity (mean - balanced):
overall = similarity.similarity()
print(f"Poses {similarity.pair_id} are {overall*100:.1f}% similar")

# Geometric mean (penalizes dissimilarity):
overall = similarity.similarity(AggregationMethod.GEOMETRIC_MEAN)
if overall > 0.80:
    print(f"Good overall match for pair {similarity.pair_id}")

# Quick pose matching check:
if similarity.matches_pose(threshold=0.8):
    print(f"Poses {similarity.pair_id} match!")

# Batch processing multiple pairs:
similarities = [
    SimilarityFeature(vals1, scores1, (1, 2)),
    SimilarityFeature(vals2, scores2, (1, 3)),
    SimilarityFeature(vals3, scores3, (2, 3)),
]

# Find best matching pair:
best = max(similarities, key=lambda s: s.similarity())
print(f"Best match: poses {best.pair_id} ({best.similarity():.2f})")

Statistical Comparison for Similarity:
---------------------------------------
Given similarity scores: [0.9, 0.9, 0.9, 0.3]
(3 matching joints, 1 non-matching)

• Mean:          0.75  → "75% similar overall"
• Geometric:     0.69  → "69% similar (penalizes mismatch)"
• Harmonic:      0.51  → "51% similar (very strict)"

Use case guidance:
- Mean:      General similarity assessment (balanced)
- Geometric: Prefer matching poses, some tolerance for variation
- Harmonic:  Require all joints to match (strict pose verification)

Comparison with SymmetryFeature:
--------------------------------
- SimilarityFeature: Compares two different poses (inter-pose)
  * Has pair_id tracking which poses are compared
  * Used for pose matching, tracking, quality assessment

- SymmetryFeature: Compares left/right within one pose (intra-pose)
  * No pair_id (single pose analysis)
  * Used for symmetry checking, pose quality

Both use same statistical methods and interpretation

Validation:
-----------
• pair_id must be tuple of two different integers
• pair_id is always ordered (smaller, larger)
• Cannot compare a pose with itself
• Example valid: (5, 8), (1, 100)
• Example invalid: (5, 5), (8, 5)

Notes:
------
- Similarity scores are in range [0.0, 1.0]
  * 1.0 = perfect similarity (angles identical)
  * 0.0 = maximum dissimilarity (angles π radians apart)
- Invalid scores are NaN (check with get_valid() before use)
- Confidence scores indicate reliability of the similarity measurement
  (minimum confidence of the two angles being compared)
- Zero similarity (complete mismatch) is replaced with TINY (1e-5) in
  geometric/harmonic means to preserve semantic meaning (penalizes score
  rather than being filtered out)
- All statistics support min_confidence filtering to ignore uncertain
  measurements
- Use geometric_mean for balanced strictness in most cases
- Use harmonic_mean for strict pose verification (e.g., exercise form checking)
=============================================================================
"""