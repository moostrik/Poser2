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


class NormalizedScalarFeature(BaseScalarFeature[FeatureEnum]):
    """Base class for normalized scalar features with statistical aggregation.

    For features with values in range [0.0, 1.0] where statistical aggregation
    is meaningful (e.g., similarity scores, symmetry scores, confidence metrics).

    Provides confidence-weighted statistics:
    - Mean, geometric mean, harmonic mean
    - Min, max, median, standard deviation
    - Filtering by minimum confidence threshold

    Subclasses must implement:
    - joint_enum(): Define the joint structure (which IntEnum to use)
    - default_range(): Must return (0.0, 1.0) for normalized features

    Design rationale:
    - Separates normalized features (where stats make sense) from
      non-normalized features like angles (where they don't)
    - All statistics are confidence-weighted for reliability
    - Can filter out low-confidence measurements
    """

    TINY: float = 1e-5  # Tiny value to replace zeros in geometric/harmonic means

    # ========== ABSTRACT METHODS ==========

    @classmethod
    @abstractmethod
    def joint_enum(cls) -> type[FeatureEnum]:
        """Returns the enum type for joints in this feature."""
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
                  min_confidence: float = 0.0) -> float:
        """Compute statistical aggregate of values with confidence filtering.

        All aggregation methods use confidence-weighted computation where
        applicable (mean, geometric mean, harmonic mean, std).

        Args:
            method: Aggregation method to use
            min_confidence: Minimum confidence to include value (default: 0.0)
                           Set higher (e.g., 0.5 or 0.7) to ignore uncertain values

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

        # Apply aggregation method
        if method == AggregationMethod.MEAN:
            # Confidence-weighted arithmetic mean
            return float(np.average(confident_values, weights=confident_scores))

        elif method == AggregationMethod.GEOMETRIC_MEAN:
            # Replace zeros with TINY (don't filter - zero has meaning!)
            safe_values = np.where(confident_values > self.TINY, confident_values, self.TINY)

            # Geometric mean in log space for numerical stability
            weighted_log_mean = np.average(np.log(safe_values), weights=confident_scores)
            return float(np.exp(weighted_log_mean))

        elif method == AggregationMethod.HARMONIC_MEAN:
            # Replace zeros with TINY (don't filter - zero has meaning!)
            safe_values = np.where(confident_values > self.TINY, confident_values, self.TINY)

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
        Useful for similarity/symmetry where all joints should score well.

        Args:
            min_confidence: Minimum confidence to include value (default: 0.0)

        Returns:
            Weighted geometric mean, or NaN if no positive values meet criteria

        Examples:
            >>> # Stricter than mean - penalizes low scores
            >>> geom = feature.geometric_mean()
            >>>
            >>> # If any joint has low similarity, overall score drops significantly
            >>> # E.g., values [0.9, 0.9, 0.2] -> geom ≈ 0.52 vs mean = 0.67
        """
        return self.aggregate(AggregationMethod.GEOMETRIC_MEAN, min_confidence)

    def harmonic_mean(self, min_confidence: float = 0.0) -> float:
        """Confidence-weighted harmonic mean of valid positive values.

        Harmonic mean heavily penalizes low values - most conservative metric.
        Useful when all joints must score well for overall success.

        Args:
            min_confidence: Minimum confidence to include value (default: 0.0)

        Returns:
            Weighted harmonic mean, or NaN if no positive values meet criteria

        Examples:
            >>> # Most conservative - heavily penalizes low scores
            >>> harm = feature.harmonic_mean()
            >>>
            >>> # E.g., values [0.9, 0.9, 0.2] -> harm ≈ 0.32 vs mean = 0.67
            >>> # Use when all joints must match well
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



"""
=============================================================================
NORMALIZEDSCALARFEATURE QUICK API REFERENCE
=============================================================================

Design Philosophy:
------------------
NormalizedScalarFeature extends BaseScalarFeature to add statistical
aggregation for features with values in [0.0, 1.0] range where statistics
are meaningful (similarity scores, symmetry scores, confidence metrics).

Key Differences from BaseScalarFeature:
- ✅ Has statistics methods (mean, geometric_mean, harmonic_mean, etc.)
- ✅ All statistics are confidence-weighted
- ✅ Can filter by minimum confidence threshold
- ✅ Only for normalized [0, 1] values

Inherited from BaseScalarFeature:
----------------------------------
Properties:
  • values: np.ndarray                             All feature values [0, 1]
  • scores: np.ndarray                             All confidence scores
  • valid_mask: np.ndarray                         Boolean validity mask
  • valid_count: int                               Number of valid values
  • len(feature): int                              Total number of joints

Single Value Access:
  • feature[joint] -> float                        Get value [0, 1]
  • feature.get(joint, fill=0.0) -> float          Get value with NaN fill
  • feature.get_value(joint, fill) -> float        Alias for get()
  • feature.get_score(joint) -> float              Get confidence score
  • feature.get_valid(joint) -> bool               Check if value is valid

Batch Operations:
  • feature.get_values(joints, fill) -> list[float]  Get multiple values
  • feature.get_scores(joints) -> list[float]        Get multiple confidences
  • feature.are_valid(joints) -> bool                Check if ALL valid

NormalizedScalarFeature-Specific:
----------------------------------
Statistical Aggregation:
  • feature.aggregate(method, min_confidence) -> float
      General-purpose aggregation with method selection

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
  • AggregationMethod.MEAN              - Arithmetic mean (balanced)
  • AggregationMethod.GEOMETRIC_MEAN    - Geometric mean (penalizes low)
  • AggregationMethod.HARMONIC_MEAN     - Harmonic mean (very strict)
  • AggregationMethod.MIN               - Minimum value
  • AggregationMethod.MAX               - Maximum value
  • AggregationMethod.MEDIAN            - Median value
  • AggregationMethod.STD               - Standard deviation

Common Usage Patterns:
----------------------
# Simple mean (all values):
avg = feature.mean()

# High-confidence mean only:
avg = feature.mean(min_confidence=0.7)

# Geometric mean (penalizes outliers):
geom = feature.geometric_mean(min_confidence=0.5)

# Using aggregate with different methods:
mean_val = feature.aggregate(AggregationMethod.MEAN)
geom_val = feature.aggregate(AggregationMethod.GEOMETRIC_MEAN)
harm_val = feature.aggregate(AggregationMethod.HARMONIC_MEAN)

# Compare different statistics:
for method in AggregationMethod:
    value = feature.aggregate(method, min_confidence=0.6)
    print(f"{method.value}: {value:.3f}")

Statistical Comparison:
-----------------------
Given values: [0.9, 0.9, 0.9, 0.2]

• Mean:          0.725  (balanced average)
• Geometric:     0.621  (penalizes the 0.2)
• Harmonic:      0.375  (heavily penalizes the 0.2)

Use case guidance:
- Mean:      General purpose, balanced
- Geometric: Need most values high, some tolerance
- Harmonic:  Need ALL values high, very strict

Notes:
------
- All methods support min_confidence filtering
- Geometric/harmonic means skip values ≤ 1e-6 (numerical stability)
- Methods return NaN if no values meet criteria
- Confidence weighting improves reliability of aggregates
- Only use for normalized [0, 1] features (not angles!)
=============================================================================
"""