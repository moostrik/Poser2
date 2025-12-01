"""
=============================================================================
SIMILARITYFEATURE API REFERENCE
=============================================================================

Concrete implementation of NormalizedScalarFeature for pose similarity assessment.

Use for: pose similarity scoring, pose quality checking, pose matching.

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
  • Similarity(values, scores)           → Direct (fast, no validation)
  • Similarity.create_empty()            → All NaN values, zero scores

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
  • len(feature): int                              Total number of elements (4)

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
  • Similarity.create_empty() -> Similarity          All NaN values, zero scores

Validation:
-----------
  • feature.validate(check_ranges=True) -> tuple[bool, str|None]
      Returns (is_valid, error_message)

Implemented Methods (from NormalizedScalarFeature):
----------------------------------------------------
Structure:
  • enum() -> type[PoseIndex]         Returns PoseIndex enum (IMPLEMENTED)

Implemented Methods (do not override):
---------------------------------------
  • range() -> tuple[float, float]         Always returns (0.0, 1.0)
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
      Best for: When most regions should be symmetric

  • feature.harmonic_mean(min_confidence=0.0) -> float
      Confidence-weighted harmonic mean (heavily penalizes low values)
      Best for: When ALL regions must be symmetric (strict matching)

  • feature.min_value(min_confidence=0.0) -> float
      Minimum symmetry value (least symmetric region)

  • feature.max_value(min_confidence=0.0) -> float
      Maximum symmetry value (most symmetric region)

  • feature.median(min_confidence=0.0) -> float
      Median symmetry value

  • feature.std(min_confidence=0.0) -> float
      Confidence-weighted standard deviation

Aggregation Methods (Enum):
----------------------------
  • AggregationMethod.MEAN              Arithmetic mean (balanced)
  • AggregationMethod.GEOMETRIC_MEAN    Geometric mean (penalizes asymmetric)
  • AggregationMethod.HARMONIC_MEAN     Harmonic mean (very strict)
  • AggregationMethod.MIN               Minimum value
  • AggregationMethod.MAX               Maximum value
  • AggregationMethod.MEDIAN            Median value
  • AggregationMethod.STD               Standard deviation

SimilarityFeature-Specific:
==========================

Similarity Methods:
-----------------
  • feature.overall_similarity(method=MEAN, min_confidence=0.0) -> float
      Compute overall similarity score using specified aggregation method
      Convenience wrapper for aggregate() with similarity-specific naming

PoseIndex Enum (tracked pose indices):
------------------------------------------------
  • POSE_0 (0)  - First tracked pose
  • POSE_1 (1)     - Second tracked pose
  • POSE_2 (2)       - Third tracked pose
  • POSE_3 (3)      - Fourth tracked pose

Statistical Comparison for Similarity:
-------------------------------------
Given similarity values: [0.9, 0.9, 0.9, 0.3]
(3 similar regions, 1 dissimilar)

• Mean:          0.75  → "75% similar overall"
• Geometric:     0.69  → "69% similar (penalizes dissimilarity)"
• Harmonic:      0.51  → "51% similar (very strict)"

Use case guidance:
- Mean:      General similarity assessment (balanced)
- Geometric: Prefer similar poses, some tolerance for variation
- Harmonic:  Require all regions similar (strict yoga/dance poses)

Comparison with SymmetryFeature:
----------------------------------
SimilarityFeature:
  • Compares current pose with multiple tracked poses (inter-pose comparison)
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
  * 1.0 = perfect similarity (left/right angles identical after mirroring)
  * 0.0 = maximum dissimilarity (left/right angles maximally different)
- Invalid values are NaN with score 0.0
- Geometric/harmonic means replace zeros with 1e-5 (numerical stability)
- Zero values have semantic meaning (complete dissimilarity) and penalize scores
- Methods return NaN if no values meet min_confidence criteria
- Confidence weighting improves reliability of aggregates
- Arrays are read-only after construction (immutable)
- Use validate() for debugging, not in production loops
- Angles should already be mirrored (right-side negated) before computing similarity
- Confidence is minimum of the two angles being compared (conservative)
=============================================================================
"""

from enum import IntEnum
from typing import cast

import numpy as np

from modules.pose.features.base.NormalizedScalarFeature import NormalizedScalarFeature, AggregationMethod


# Module-level configuration (set once at app startup)
_PoseEnum: type[IntEnum] | None = None


def configure_similarity(max_poses: int) -> None:
    """Configure Similarity feature with number of poses to track.

    Must be called once at application initialization before creating any Frame instances.

    Args:
        max_poses: Maximum number of poses to compare similarities for
    """
    global _PoseEnum
    if _PoseEnum is None:
        _PoseEnum = cast(type[IntEnum], IntEnum("PoseIndex", {f"POSE_{i}": i for i in range(max_poses)}))


class Similarity(NormalizedScalarFeature):
    """Similarity scores between current pose and multiple tracked poses."""

    def __init__(self, values: np.ndarray, scores: np.ndarray) -> None:
        if _PoseEnum is None:
            raise RuntimeError(
                "Similarity not configured. Call configure_similarity(max_poses) at app startup."
            )
        super().__init__(values, scores)

    @classmethod
    def enum(cls) -> type[IntEnum]:
        if _PoseEnum is None:
            raise RuntimeError(
                "Similarity not configured. Call configure_similarity(max_poses) at app startup."
            )
        return _PoseEnum

    def overall_similarity(self) -> float:
        """Compute overall similarity using harmonic mean (strict matching)."""
        return self.aggregate(method=AggregationMethod.HARMONIC_MEAN, min_confidence=0.0, exponent=2.0)
