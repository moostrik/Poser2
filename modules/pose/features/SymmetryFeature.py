"""
=============================================================================
SYMMETRYFEATURE API REFERENCE
=============================================================================

Concrete implementation of NormalizedScalarFeature for left/right body symmetry.

Use for: pose symmetry assessment, pose quality checking, symmetric pose detection.

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
  • SymmetryFeature(values, scores)           → Direct (fast, no validation)
  • SymmetryFeature.create_empty()            → All NaN values, zero scores

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
  • A scalar symmetry value (float) in [0.0, 1.0] - may be NaN for invalid/missing data
  • A confidence score [0.0, 1.0]

Storage:
  • values: np.ndarray, shape (n_elements,), dtype float32, range [0.0, 1.0]
  • scores: np.ndarray, shape (n_elements,), dtype float32

Properties:
-----------
  • values: np.ndarray                             All symmetry values (n_elements,)
  • scores: np.ndarray                             All confidence scores (n_elements,)
  • valid_mask: np.ndarray                         Boolean validity mask (n_elements,)
  • valid_count: int                               Number of valid values
  • len(feature): int                              Total number of elements (4)

Single Value Access:
--------------------
  • feature[element] -> float                      Get symmetry (supports enum or int)
  • feature.get(element, fill=np.nan) -> float     Get symmetry with NaN handling
  • feature.get_value(element, fill) -> float      Alias for get()
  • feature.get_score(element) -> float            Get confidence score
  • feature.get_valid(element) -> bool             Check if symmetry is valid

Batch Operations:
-----------------
  • feature.get_values(elements, fill) -> list[float]  Get multiple symmetries
  • feature.get_scores(elements) -> list[float]        Get multiple scores
  • feature.are_valid(elements) -> bool                Check if ALL valid

Factory Methods:
----------------
  • SymmetryFeature.create_empty() -> SymmetryFeature          All NaN values, zero scores

Validation:
-----------
  • feature.validate(check_ranges=True) -> tuple[bool, str|None]
      Returns (is_valid, error_message)

Implemented Methods (from NormalizedScalarFeature):
----------------------------------------------------
Structure:
  • feature_enum() -> type[SymmetryRegion]         Returns SymmetryRegion enum (IMPLEMENTED)

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

SymmetryFeature-Specific:
==========================

Symmetry Methods:
-----------------
  • feature.overall_symmetry(method=MEAN, min_confidence=0.0) -> float
      Compute overall symmetry score using specified aggregation method
      Convenience wrapper for aggregate() with symmetry-specific naming

SymmetryRegion Enum (symmetric landmark pairs):
------------------------------------------------
  • shoulder (0)  - Left/right shoulder pair symmetry
  • elbow (1)     - Left/right elbow pair symmetry
  • hip (2)       - Left/right hip pair symmetry
  • knee (3)      - Left/right knee pair symmetry

Statistical Comparison for Symmetry:
-------------------------------------
Given symmetry values: [0.9, 0.9, 0.9, 0.3]
(3 symmetric regions, 1 asymmetric)

• Mean:          0.75  → "75% symmetric overall"
• Geometric:     0.69  → "69% symmetric (penalizes asymmetry)"
• Harmonic:      0.51  → "51% symmetric (very strict)"

Use case guidance:
- Mean:      General symmetry assessment (balanced)
- Geometric: Prefer symmetric poses, some tolerance for variation
- Harmonic:  Require all regions symmetric (strict yoga/dance poses)

Comparison with SimilarityFeature:
----------------------------------
SymmetryFeature:
  • Compares left/right within one pose (intra-pose comparison)
  • No pair_id (single pose analysis)
  • Used for symmetry checking, pose quality

SimilarityFeature:
  • Compares two different poses (inter-pose comparison)
  • Has pair_id tracking which poses are compared
  • Used for pose matching, tracking, quality assessment

Both use same statistical methods and interpretation.

Notes:
------
- Symmetry values are in range [0.0, 1.0]
  * 1.0 = perfect symmetry (left/right angles identical after mirroring)
  * 0.0 = maximum asymmetry (left/right angles maximally different)
- Invalid values are NaN with score 0.0
- Geometric/harmonic means replace zeros with 1e-5 (numerical stability)
- Zero values have semantic meaning (complete asymmetry) and penalize scores
- Methods return NaN if no values meet min_confidence criteria
- Confidence weighting improves reliability of aggregates
- Arrays are read-only after construction (immutable)
- Use validate() for debugging, not in production loops
- Angles should already be mirrored (right-side negated) before computing symmetry
- Confidence is minimum of the two angles being compared (conservative)
=============================================================================
"""

from enum import IntEnum

from modules.pose.features.base.BaseFeature import NORMALIZED_RANGE
from modules.pose.features.base.NormalizedScalarFeature import NormalizedScalarFeature, AggregationMethod
from modules.pose.features.AngleFeature import AngleLandmark


class SymmetryElement(IntEnum):
    """Symmetric landmark pairs for body symmetry measurement.

    Each value represents a pair of left/right landmarks that should
    exhibit similar angles in symmetric poses.
    """
    shoulder = 0
    elbow = 1
    hip = 2
    knee = 3


# Constants
SYMMETRY_ELEMENT_NAMES: list[str] = [e.name for e in SymmetryElement]
SYMMETRY_NUM_ELEMENTS: int = len(SymmetryElement)  # for backward compatibility

# Maps each symmetric landmark type to its left/right AngleLandmark pair
_SYMMETRY_PAIRS: dict[SymmetryElement, tuple[AngleLandmark, AngleLandmark]] = {
    SymmetryElement.shoulder: (AngleLandmark.left_shoulder, AngleLandmark.right_shoulder),
    SymmetryElement.elbow: (AngleLandmark.left_elbow, AngleLandmark.right_elbow),
    SymmetryElement.hip: (AngleLandmark.left_hip, AngleLandmark.right_hip),
    SymmetryElement.knee: (AngleLandmark.left_knee, AngleLandmark.right_knee),
}

SYMMETRY_RANGE: tuple[float, float] = NORMALIZED_RANGE


class SymmetryFeature(NormalizedScalarFeature[SymmetryElement]):
    """Symmetry scores for left/right landmark pairs (range [0, 1]).

    Measures how similar left/right landmark angles are after mirroring.

    Values:
    - 1.0 = perfect symmetry (left/right angles identical after mirroring)
    - 0.0 = maximum asymmetry (left/right angles completely different)

    Each symmetry score has a confidence based on the minimum confidence
    of the two angles being compared.

    Inherits statistical aggregation methods from NormalizedScalarFeature:
    - mean(), geometric_mean(), harmonic_mean()
    - aggregate() with multiple methods
    - All methods support confidence filtering

    Note: Angles should already be mirrored (right-side negated) before
          computing symmetry.
    """

    # ========== ABSTRACT METHOD IMPLEMENTATIONS ==========

    @classmethod
    def feature_enum(cls) -> type[SymmetryElement]:
        """Returns SymmetricRegion enum."""
        return SymmetryElement

    # ========== SYMMETRY-SPECIFIC CONVENIENCE METHODS ==========

    def overall_symmetry(self, method: AggregationMethod = AggregationMethod.MEAN,
                        min_confidence: float = 0.0) -> float:
        """Compute overall symmetry score using specified aggregation method.

        Args:
            method: Statistical aggregation method (default: MEAN)
            min_confidence: Minimum confidence to include landmark (default: 0.0)

        Returns:
            Overall symmetry score [0, 1], or NaN if no landmarks meet criteria

        Examples:
            >>> # Mean symmetry (default, balanced)
            >>> overall = symmetry.overall_symmetry()
            >>>
            >>> # Geometric mean (penalizes low symmetry more)
            >>> overall = symmetry.overall_symmetry(AggregationMethod.GEOMETRIC_MEAN)
            >>>
            >>> # Harmonic mean (very strict - heavily penalizes asymmetry)
            >>> overall = symmetry.overall_symmetry(AggregationMethod.HARMONIC_MEAN)
            >>>
            >>> # Only trust high-confidence measurements
            >>> overall = symmetry.overall_symmetry(
            ...     AggregationMethod.MEAN,
            ...     min_confidence=0.7
            ... )
            >>>
            >>> # For strict symmetric poses (all landmarks must be symmetric):
            >>> overall = symmetry.overall_symmetry(
            ...     AggregationMethod.HARMONIC_MEAN,
            ...     min_confidence=0.6
            ... )
        """
        return self.aggregate(method, min_confidence)
