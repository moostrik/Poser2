"""
=============================================================================
NORMALIZEDSCALARFEATURE API REFERENCE
=============================================================================

Extends BaseScalarFeature with statistical aggregation for normalized [0, 1] features.

Use for: similarity scores, symmetry scores, confidence metrics, normalized ratios.

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
  • MyFeature(values, scores)           → Direct (fast, no validation)
  • MyFeature.create_empty()            → All NaN values, zero scores

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
  • A scalar value (float) in [0.0, 1.0] - may be NaN for invalid/missing data
  • A confidence score [0.0, 1.0]

Storage:
  • values: np.ndarray, shape (n_elements,), dtype float32, range [0.0, 1.0]
  • scores: np.ndarray, shape (n_elements,), dtype float32

Properties:
-----------
  • values: np.ndarray                             All scalar values (n_elements,)
  • scores: np.ndarray                             All confidence scores (n_elements,)
  • valid_mask: np.ndarray                         Boolean validity mask (n_elements,)
  • valid_count: int                               Number of valid values
  • len(feature): int                              Total number of elements

Single Value Access:
--------------------
  • feature[element] -> float                      Get value (supports enum or int)
  • feature.get(element, fill=np.nan) -> float     Get value with NaN handling
  • feature.get_value(element, fill) -> float      Alias for get()
  • feature.get_score(element) -> float            Get confidence score
  • feature.get_valid(element) -> bool             Check if value is valid

Batch Operations:
-----------------
  • feature.get_values(elements, fill) -> list[float]  Get multiple values
  • feature.get_scores(elements) -> list[float]        Get multiple scores
  • feature.are_valid(elements) -> bool                Check if ALL valid

Factory Methods:
----------------
  • MyFeature.create_empty() -> MyFeature          All NaN values, zero scores

Validation:
-----------
  • feature.validate(check_ranges=True) -> tuple[bool, str|None]
      Returns (is_valid, error_message)

Abstract Methods (must implement in subclasses):
-------------------------------------------------
  • feature_enum() -> type[IntEnum]                Feature element enum

Implemented Methods (do not override):
---------------------------------------
  • default_range() -> tuple[float, float]         Always returns (0.0, 1.0)


NormalizedScalarFeature-Specific:
==================================

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
      Best for: When all values should be high (similarity matching)

  • feature.harmonic_mean(min_confidence=0.0) -> float
      Confidence-weighted harmonic mean (heavily penalizes low values)
      Best for: When all values must be high (strict matching)

  • feature.min_value(min_confidence=0.0) -> float
      Minimum value meeting confidence threshold

  • feature.max_value(min_confidence=0.0) -> float
      Maximum value meeting confidence threshold

  • feature.median(min_confidence=0.0) -> float
      Median value meeting confidence threshold

  • feature.std(min_confidence=0.0) -> float
      Confidence-weighted standard deviation

Aggregation Methods (Enum):
----------------------------
  • AggregationMethod.MEAN              Arithmetic mean (balanced)
  • AggregationMethod.GEOMETRIC_MEAN    Geometric mean (penalizes low)
  • AggregationMethod.HARMONIC_MEAN     Harmonic mean (very strict)
  • AggregationMethod.MIN               Minimum value
  • AggregationMethod.MAX               Maximum value
  • AggregationMethod.MEDIAN            Median value
  • AggregationMethod.STD               Standard deviation

Statistical Comparison:
-----------------------
Given values: [0.9, 0.9, 0.9, 0.2]

• Mean:          0.725  (balanced average)
• Geometric:     0.621  (penalizes the 0.2)
• Harmonic:      0.375  (heavily penalizes the 0.2)

Use case guidance:
- Mean:      General purpose, balanced
- Geometric: Need most values high, some tolerance for outliers
- Harmonic:  Need ALL values high, very strict

Notes:
------
- All values must be in [0.0, 1.0] range
- Invalid values are NaN with score 0.0
- Geometric/harmonic means replace zeros with 1e-5 (numerical stability)
- Zero values have semantic meaning (failure) and penalize scores
- Methods return NaN if no values meet min_confidence criteria
- Confidence weighting improves reliability of aggregates
- Arrays are read-only after construction (immutable)
- Use validate() for debugging, not in production loops
- Only use for normalized features (not angles or unbounded values!)
=============================================================================
"""

from abc import abstractmethod
from enum import Enum

import numpy as np

from modules.pose.features.base.BaseScalarFeature import BaseScalarFeature, FeatureEnum


class AggregationMethod(Enum):
    """Statistical aggregation methods for normalized scalar features.

    All methods support confidence-weighted computation and filtering
    by minimum confidence threshold.
    """
    MEAN = 'mean'
    GEOMETRIC_MEAN = 'geometric_mean'
    HARMONIC_MEAN = 'harmonic_mean'
    MIN = 'min'
    MAX = 'max'
    MEDIAN = 'median'
    STD = 'std'


_TINY: float = 1e-5  # Tiny value to replace zeros in geometric/harmonic means

class NormalizedScalarFeature(BaseScalarFeature[FeatureEnum]):
    """Base class for normalized scalar features with statistical aggregation.

    For features with values in range [0.0, 1.0] where statistical aggregation
    is meaningful (e.g., similarity scores, symmetry scores, confidence metrics).

    Provides confidence-weighted statistics:
    - Mean, geometric mean, harmonic mean
    - Min, max, median, standard deviation
    - Filtering by minimum confidence threshold

    Subclasses must implement:
    - feature_enum(): Define the element structure (which IntEnum to use)
    - default_range(): Must return (0.0, 1.0) for normalized features

    Design rationale:
    - Separates normalized features (where stats make sense) from
      non-normalized features like angles (where they don't)
    - All statistics are confidence-weighted for reliability
    - Can filter out low-confidence measurements
    """

    # ========== ABSTRACT METHODS ==========

    @classmethod
    @abstractmethod
    def feature_enum(cls) -> type[FeatureEnum]:
        """Returns the enum type for elements in this feature."""
        pass

    @classmethod
    def default_range(cls) -> tuple[float, float]:
        """Returns (0.0, 1.0) for normalized features.

        All subclasses must have values in [0, 1] range.
        Do not override this method.
        """
        return (0.0, 1.0)

    # ========== STATISTICS (CONFIDENCE-WEIGHTED) ==========

    def aggregate(self, method: AggregationMethod = AggregationMethod.MEAN,
                  min_confidence: float = 0.0,
                  exponent: float = 1.0) -> float:
        """
        Compute statistical aggregate of values with confidence filtering and optional exponentiation.

        Args:
            method: Aggregation method to use
            min_confidence: Minimum confidence to include value (default: 0.0)
            exponent: Exponent to apply to each value before aggregation (default: 1.0)

        Returns:
            Aggregated value, or NaN if no values meet criteria

        Note:
            For geometric and harmonic means, zero values are replaced with
            a tiny value (1e-6) rather than filtered out, because zero has
            semantic meaning (complete failure/mismatch) and should penalize
            the overall score rather than being ignored.
        """
        # Filter by confidence threshold and validity
        confident_mask = (self._scores >= min_confidence) & self._valid_mask

        if not np.any(confident_mask):
            return np.nan

        confident_values = self._values[confident_mask]
        confident_scores = self._scores[confident_mask]

        # Apply exponent to each value before aggregation
        if exponent != 1.0:
            confident_values = confident_values ** exponent

        # Apply aggregation method
        if method == AggregationMethod.MEAN:
            # Confidence-weighted arithmetic mean
            return float(np.average(confident_values))

        elif method == AggregationMethod.GEOMETRIC_MEAN:
            # Replace zeros with TINY (don't filter - zero has meaning!)
            safe_values = np.where(confident_values > _TINY, confident_values, _TINY)

            # Geometric mean in log space for numerical stability
            weighted_log_mean = np.average(np.log(safe_values))
            return float(np.exp(weighted_log_mean))

        elif method == AggregationMethod.HARMONIC_MEAN:
            # Replace zeros with TINY (don't filter - zero has meaning!)
            safe_values = np.where(confident_values > _TINY, confident_values, _TINY)

            # Weighted harmonic mean: sum(w) / sum(w/x)
            return float(len(confident_values) / np.sum(1.0 / confident_values))

        elif method == AggregationMethod.MIN:
            # Minimum value (no weighting applicable)
            return float(np.min(confident_values))

        elif method == AggregationMethod.MAX:
            # Maximum value (no weighting applicable)
            return float(np.max(confident_values))

        elif method == AggregationMethod.MEDIAN:
            # Median (no weighting - numpy median doesn't support weights)
            return float(np.median(confident_values))

        elif method == AggregationMethod.STD:
            # Confidence-weighted standard deviation
            if len(confident_values) < 2:
                return np.nan

            # Weighted variance formula: sum(w * (x - mean)^2) / sum(w)
            weighted_mean = np.average(confident_values)
            variance = np.average((confident_values - weighted_mean) ** 2,)
            return float(np.sqrt(variance))

        else:
            raise ValueError(f"Unknown aggregation method: {method}")

    def aggregate_weighted(self, method: AggregationMethod = AggregationMethod.MEAN,
                  min_confidence: float = 0.0,
                  exponent: float = 1.0) -> float:
        """
        Compute statistical aggregate of values with confidence filtering and optional exponentiation.

        Args:
            method: Aggregation method to use
            min_confidence: Minimum confidence to include value (default: 0.0)
            exponent: Exponent to apply to each value before aggregation (default: 1.0)

        Returns:
            Aggregated value, or NaN if no values meet criteria

        Note:
            For geometric and harmonic means, zero values are replaced with
            a tiny value (1e-6) rather than filtered out, because zero has
            semantic meaning (complete failure/mismatch) and should penalize
            the overall score rather than being ignored.
        """
        # Filter by confidence threshold and validity
        confident_mask = (self._scores >= min_confidence) & self._valid_mask

        if not np.any(confident_mask):
            return np.nan

        confident_values = self._values[confident_mask]
        confident_scores = self._scores[confident_mask]

        # Apply exponent to each value before aggregation
        if exponent != 1.0:
            confident_values = confident_values ** exponent

        # Apply aggregation method
        if method == AggregationMethod.MEAN:
            # Confidence-weighted arithmetic mean
            return float(np.average(confident_values, weights=confident_scores))

        elif method == AggregationMethod.GEOMETRIC_MEAN:
            # Replace zeros with TINY (don't filter - zero has meaning!)
            safe_values = np.where(confident_values > _TINY, confident_values, _TINY)

            # Geometric mean in log space for numerical stability
            weighted_log_mean = np.average(np.log(safe_values), weights=confident_scores)
            return float(np.exp(weighted_log_mean))

        elif method == AggregationMethod.HARMONIC_MEAN:
            # Replace zeros with TINY (don't filter - zero has meaning!)
            safe_values = np.where(confident_values > _TINY, confident_values, _TINY)

            # Weighted harmonic mean: sum(w) / sum(w/x)
            return float(np.sum(confident_scores) / np.sum(confident_scores / safe_values))

        elif method == AggregationMethod.MIN:
            # Minimum value (no weighting applicable)
            return float(np.min(confident_values))

        elif method == AggregationMethod.MAX:
            # Maximum value (no weighting applicable)
            return float(np.max(confident_values))

        elif method == AggregationMethod.MEDIAN:
            # Median (no weighting - numpy median doesn't support weights)
            return float(np.median(confident_values))

        elif method == AggregationMethod.STD:
            # Confidence-weighted standard deviation
            if len(confident_values) < 2:
                return np.nan

            # Weighted variance formula: sum(w * (x - mean)^2) / sum(w)
            weighted_mean = np.average(confident_values, weights=confident_scores)
            variance = np.average(
                (confident_values - weighted_mean) ** 2,
                weights=confident_scores
            )
            return float(np.sqrt(variance))

        else:
            raise ValueError(f"Unknown aggregation method: {method}")

    # ========== CONVENIENCE METHODS ==========

    def mean(self, min_confidence: float = 0.0) -> float:
        """Confidence-weighted mean of valid values.

        Args:
            min_confidence: Minimum confidence to include value (default: 0.0)

        Returns:
            Weighted mean, or NaN if no values meet criteria

        Examples:
            >>> # All valid values
            >>> avg = feature.mean()
            >>>
            >>> # Only high-confidence values
            >>> avg = feature.mean(min_confidence=0.7)
        """
        return self.aggregate(AggregationMethod.MEAN, min_confidence)

    def geometric_mean(self, min_confidence: float = 0.0) -> float:
        """Confidence-weighted geometric mean of valid positive values.

        Geometric mean penalizes low values more than arithmetic mean.
        Useful for similarity/symmetry where all elements should score well.

        Args:
            min_confidence: Minimum confidence to include value (default: 0.0)

        Returns:
            Weighted geometric mean, or NaN if no positive values meet criteria

        Examples:
            >>> # Stricter than mean - penalizes low scores
            >>> geom = feature.geometric_mean()
            >>>
            >>> # If any element has low similarity, overall score drops significantly
            >>> # E.g., values [0.9, 0.9, 0.2] -> geom ≈ 0.52 vs mean = 0.67
        """
        return self.aggregate(AggregationMethod.GEOMETRIC_MEAN, min_confidence)

    def harmonic_mean(self, min_confidence: float = 0.0) -> float:
        """Confidence-weighted harmonic mean of valid positive values.

        Harmonic mean heavily penalizes low values - most conservative metric.
        Useful when all elements must score well for overall success.

        Args:
            min_confidence: Minimum confidence to include value (default: 0.0)

        Returns:
            Weighted harmonic mean, or NaN if no positive values meet criteria

        Examples:
            >>> # Most conservative - heavily penalizes low scores
            >>> harm = feature.harmonic_mean()
            >>>
            >>> # E.g., values [0.9, 0.9, 0.2] -> harm ≈ 0.32 vs mean = 0.67
            >>> # Use when all elements must match well
        """
        return self.aggregate(AggregationMethod.HARMONIC_MEAN, min_confidence)

    def min_value(self, min_confidence: float = 0.0) -> float:
        """Minimum of values meeting confidence threshold.

        Args:
            min_confidence: Minimum confidence to include value (default: 0.0)

        Returns:
            Minimum value, or NaN if no values meet criteria
        """
        return self.aggregate(AggregationMethod.MIN, min_confidence)

    def max_value(self, min_confidence: float = 0.0) -> float:
        """Maximum of values meeting confidence threshold.

        Args:
            min_confidence: Minimum confidence to include value (default: 0.0)

        Returns:
            Maximum value, or NaN if no values meet criteria
        """
        return self.aggregate(AggregationMethod.MAX, min_confidence)

    def median(self, min_confidence: float = 0.0) -> float:
        """Median of values meeting confidence threshold.

        Args:
            min_confidence: Minimum confidence to include value (default: 0.0)

        Returns:
            Median value, or NaN if no values meet criteria
        """
        return self.aggregate(AggregationMethod.MEDIAN, min_confidence)

    def std(self, min_confidence: float = 0.0) -> float:
        """Confidence-weighted standard deviation.

        Args:
            min_confidence: Minimum confidence to include value (default: 0.0)

        Returns:
            Weighted standard deviation, or NaN if fewer than 2 values meet criteria
        """
        return self.aggregate(AggregationMethod.STD, min_confidence)
