"""
=============================================================================
SIMILARITYFEATURE API REFERENCE
=============================================================================

Concrete implementation of NormalizedScalarFeature for pose-to-pose similarity.

Use for: pose matching, pose tracking, quality assessment, pose comparison.

Summary of BaseFeature Design Philosophy:
==========================================

Immutability & Ownership:
  • Features are IMMUTABLE - arrays set to read-only after construction
  • Constructor takes OWNERSHIP - caller must not modify arrays after passing
  • Modifications create new features (functional style)

Data Access (two patterns):
  Raw (numpy):     feature.values, feature.scores, feature[element]
  Python-friendly: feature.get(element, fill), feature.get_score(element)

NaN Semantics:
  • Invalid data = NaN with score 0.0 (enforced)
  • Use get(element, fill=0.0) for automatic NaN handling

Cached Properties:
  • Subclasses may add @cached_property (safe due to immutability)

Construction:
  • SimilarityFeature(values, scores, pair_id)     → Direct (fast, no validation)
  • SimilarityFeature.create_empty()               → All NaN values, zero scores
  • SimilarityFeature.from_values(values, ...)     → Auto-generate scores if None
  • SimilarityFeature.create_validated(...)        → Full validation, raises on error

Validation:
  • Asserts in constructors (removed with -O flag for production)
  • validate() method for debugging/testing/untrusted input
  • Fast by default, validate only when needed

Performance:
  Fast:     Property access, indexing, cached properties, array ops
  Moderate: get(), get_score() (Python conversion)
  Slow:     get_values(), get_scores() (iteration), validate()

Inherited from BaseScalarFeature:
==================================

Structure:
----------
Each element has:
  • A scalar similarity value (float) in [0.0, 1.0] - may be NaN for invalid/missing data
  • A confidence score [0.0, 1.0]

Storage:
  • values: np.ndarray, shape (n_elements,), dtype float32, range [0.0, 1.0]
  • scores: np.ndarray, shape (n_elements,), dtype float32

Properties:
-----------
  • values: np.ndarray                             All similarity values (n_elements,)
  • scores: np.ndarray                             All confidence scores (n_elements,)
  • valid_mask: np.ndarray                         Boolean validity mask (n_elements,)
  • valid_count: int                               Number of valid values
  • len(feature): int                              Total number of elements (9)

Single Value Access:
--------------------
  • feature[element] -> float                      Get similarity (supports enum or int)
  • feature.get(element, fill=np.nan) -> float     Get similarity with NaN handling
  • feature.get_value(element, fill) -> float      Alias for get()
  • feature.get_score(element) -> float            Get confidence score
  • feature.get_valid(element) -> bool             Check if similarity is valid

Batch Operations:
-----------------
  • feature.get_values(elements, fill) -> list[float]  Get multiple similarities
  • feature.get_scores(elements) -> list[float]        Get multiple scores
  • feature.are_valid(elements) -> bool                Check if ALL valid

Factory Methods:
----------------
  • SimilarityFeature.create_empty() -> SimilarityFeature          All NaN values, zero scores
  • SimilarityFeature.from_values(values, scores?) -> SimilarityFeature  Auto-generate scores if None
  • SimilarityFeature.create_validated(values, scores) -> SimilarityFeature  Full validation

Validation:
-----------
  • feature.validate(check_ranges=True) -> tuple[bool, str|None]
      Returns (is_valid, error_message)

Implemented Methods (from NormalizedScalarFeature):
----------------------------------------------------
Structure:
  • feature_enum() -> type[AngleLandmark]          Returns AngleLandmark enum (IMPLEMENTED)

Implemented Methods (do not override):
---------------------------------------
  • default_range() -> tuple[float, float]         Always returns (0.0, 1.0)
                                                   (Already implemented in NormalizedScalarFeature)

Inherited from NormalizedScalarFeature:
========================================

Statistical Aggregation:
------------------------
All methods support:
  • Confidence-weighted computation
  • Filtering by minimum confidence threshold
  • Return NaN if no values meet criteria

Core Methods:
  • feature.aggregate(method, min_confidence=0.0) -> float
      General aggregation with method selection

  • feature.mean(min_confidence=0.0) -> float
      Confidence-weighted arithmetic mean
      Best for: General purpose averaging

  • feature.geometric_mean(min_confidence=0.0) -> float
      Confidence-weighted geometric mean (penalizes low values)
      Best for: When most elements should be similar

  • feature.harmonic_mean(min_confidence=0.0) -> float
      Confidence-weighted harmonic mean (heavily penalizes low values)
      Best for: When ALL elements must be similar (strict matching)

  • feature.min_value(min_confidence=0.0) -> float
      Minimum similarity value (worst matching element)

  • feature.max_value(min_confidence=0.0) -> float
      Maximum similarity value (best matching element)

  • feature.median(min_confidence=0.0) -> float
      Median similarity value

  • feature.std(min_confidence=0.0) -> float
      Confidence-weighted standard deviation

Aggregation Methods (Enum):
----------------------------
  • AggregationMethod.MEAN              Arithmetic mean (balanced)
  • AggregationMethod.GEOMETRIC_MEAN    Geometric mean (penalizes dissimilar)
  • AggregationMethod.HARMONIC_MEAN     Harmonic mean (very strict)
  • AggregationMethod.MIN               Minimum value
  • AggregationMethod.MAX               Maximum value
  • AggregationMethod.MEDIAN            Median value
  • AggregationMethod.STD               Standard deviation

SimilarityFeature-Specific:
============================

Pair Tracking:
--------------
  • feature.pair_id -> tuple[int, int]             Tuple (id_1, id_2) identifying the two poses
                                                   Always ordered as (smaller, larger)
  • feature.id_1 -> int                            First pose ID (always smaller)
  • feature.id_2 -> int                            Second pose ID (always larger)

Similarity Methods:
-------------------
  • feature.similarity(method=MEAN, min_confidence=0.0) -> float
      Compute overall similarity score using specified aggregation method
      Convenience wrapper for aggregate() with similarity-specific naming

  • feature.matches_pose(threshold=0.8, method=GEOMETRIC_MEAN, min_confidence=0.0) -> bool
      Check if poses match based on overall similarity threshold
      Returns True if similarity >= threshold, False otherwise

AngleLandmark Enum (used for similarity elements):
---------------------------------------------------
  • left_shoulder (0)    - Shoulder angle similarity
  • right_shoulder (1)   - Shoulder angle similarity
  • left_elbow (2)       - Elbow angle similarity
  • right_elbow (3)      - Elbow angle similarity
  • left_hip (4)         - Hip angle similarity
  • right_hip (5)        - Hip angle similarity
  • left_knee (6)        - Knee angle similarity
  • right_knee (7)       - Knee angle similarity
  • head (8)             - Head angle similarity

Statistical Comparison for Similarity:
---------------------------------------
Given similarity values: [0.9, 0.9, 0.9, 0.3]
(3 matching elements, 1 non-matching)

• Mean:          0.75  → "75% similar overall"
• Geometric:     0.69  → "69% similar (penalizes mismatch)"
• Harmonic:      0.51  → "51% similar (very strict)"

Use case guidance:
- Mean:      General similarity assessment (balanced)
- Geometric: Prefer matching poses, some tolerance for variation
- Harmonic:  Require all elements to match (strict pose verification)

Comparison with SymmetryFeature:
--------------------------------
SimilarityFeature:
  • Compares two different poses (inter-pose comparison)
  • Has pair_id tracking which poses are compared
  • Used for pose matching, tracking, quality assessment

SymmetryFeature:
  • Compares left/right within one pose (intra-pose comparison)
  • No pair_id (single pose analysis)
  • Used for symmetry checking, pose quality

Both use same statistical methods and interpretation.

Notes:
------
- Similarity values are in range [0.0, 1.0]
  * 1.0 = perfect similarity (angles identical)
  * 0.0 = maximum dissimilarity (angles π radians apart)
- Invalid values are NaN with score 0.0
- Geometric/harmonic means replace zeros with 1e-5 (numerical stability)
- Zero values have semantic meaning (complete mismatch) and penalize scores
- Methods return NaN if no values meet min_confidence criteria
- Confidence weighting improves reliability of aggregates
- Arrays are read-only after construction (immutable)
- Use validate() for debugging, not in production loops
- pair_id validation uses asserts (removed with -O flag for production)
- pair_id must be tuple of two different integers, always ordered (smaller, larger)
- Cannot compare a pose with itself (enforced by assert in development)

=============================================================================

SIMILARITYBATCH API REFERENCE
=============================================================================

Collection of all similarity features for multiple pose pairs in a frame.

Structure:
----------
  • similarities: list[SimilarityFeature]          List of all pair comparisons
  • timestamp: float                               When this batch was created
  • len(batch): int                                Number of pairs in batch
  • batch.is_empty: bool                           True if no pairs

Lookup & Iteration:
-------------------
  • batch.get_pair(pair_id) -> SimilarityFeature | None
      O(1) lookup for specific pair
      pair_id normalized to (min, max) automatically

  • (pair_id) in batch -> bool
      Check if pair exists in batch

  • for similarity in batch:
      Iterate all pairs

Top Matches:
------------
  • batch.get_top_pairs(n=5, min_valid_elements=1, method=MEAN, min_confidence=0.0) -> list[SimilarityFeature]
      Get top N most similar pairs, sorted by similarity (descending)

Notes:
------
- Immutable dataclass (frozen=True)
- O(1) pair lookup via internal dict
- pair_id automatically normalized to (min, max) for lookups
- Top pairs filtering supports minimum valid element count and confidence thresholds
- Empty batch check with is_empty property
=============================================================================
"""

import time
from dataclasses import dataclass, field
from typing import Iterator, Optional, Callable

import numpy as np

from modules.pose.features.base.BaseFeature import NORMALIZED_RANGE
from modules.pose.features.base.NormalizedScalarFeature import NormalizedScalarFeature, AggregationMethod
from modules.pose.features.AngleFeature import AngleLandmark


# Reuse AngleLandmark enum for similarity (same landmarks as angles)
# No need to define a new enum since similarity is measured per angle landmark

# Constants
SIMILARITY_RANGE: tuple[float, float] = NORMALIZED_RANGE


class SimilarityFeature(NormalizedScalarFeature[AngleLandmark]):
    """Similarity scores between two poses for all angle landmarks (range [0, 1]).

    Measures how similar corresponding angle landmarks are between two poses.
    Values are similarity scores [0, 1] per landmark:
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
            values: Similarity scores [0, 1] for each element
            scores: Confidence scores for each similarity measurement
            pair_id: Tuple of (pose_id_1, pose_id_2) being compared
        """
        # Validate pair_id (asserts for development, removed with -O)
        assert pair_id[0] != pair_id[1], f"Cannot compare pose with itself: {pair_id[0]}"
        assert pair_id[0] < pair_id[1], f"pair_id must be ordered as (smaller, larger), got {pair_id}"

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
        """Tuple (id_1, id_2) identifying the two poses being compared (always ordered as smaller, larger)."""
        return self._pair_id

    @property
    def id_1(self) -> int:
        """First pose ID (always the smaller ID)."""
        return self._pair_id[0]

    @property
    def id_2(self) -> int:
        """Second pose ID (always the larger ID)."""
        return self._pair_id[1]

    # ========== SIMILARITY-SPECIFIC CONVENIENCE METHODS ==========

    def similarity(self, method: AggregationMethod = AggregationMethod.MEAN,
                          min_confidence: float = 0.0) -> float:
        """Compute overall similarity score using specified aggregation method.

        Args:
            method: Statistical aggregation method (default: MEAN)
            min_confidence: Minimum confidence to include element (default: 0.0)

        Returns:
            Overall similarity score [0, 1], or NaN if no elements meet criteria

        Examples:
            >>> # Mean similarity (default, balanced)
            >>> overall = similarity.similarity()
            >>>
            >>> # Geometric mean (penalizes dissimilar landmarks more)
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
            >>> # For strict pose matching (all landmarks must match):
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
            min_confidence: Minimum confidence to include element (default: 0.0)

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
        min_valid_landmarks: int = 1,
        method: AggregationMethod = AggregationMethod.MEAN,
        min_confidence: float = 0.0
    ) -> list[SimilarityFeature]:
        """Get top N most similar pairs, sorted by similarity (descending).

        Args:
            n: Number of top pairs to return (default: 5)
            min_valid_landmarks: Minimum valid elements required (default: 1)
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
            ...     min_valid_landmarks=10,
            ...     method=AggregationMethod.GEOMETRIC_MEAN,
            ...     min_confidence=0.7
            ... )
        """
        if self.is_empty or n <= 0:
            return []

        # Filter by minimum valid landmarks
        valid_pairs = [
            sim for sim in self.similarities
            if sim.valid_count >= min_valid_landmarks
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
