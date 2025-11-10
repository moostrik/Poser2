from enum import IntEnum

from modules.pose.features.base.BaseFeature import NORMALIZED_RANGE
from modules.pose.features.base.NormalizedScalarFeature import NormalizedScalarFeature, AggregationMethod
from modules.pose.features.AngleFeature import AngleLandmark


class SymmetricJoint(IntEnum):
    """Symmetric joint pairs for body symmetry measurement.

    Each value represents a pair of left/right joints that should
    exhibit similar angles in symmetric poses.
    """
    shoulder = 0
    elbow = 1
    hip = 2
    knee = 3


# Constants
SYMM_JOINT_NAMES: list[str] = [e.name for e in SymmetricJoint]
SYMM_NUM_JOINTS: int = len(SymmetricJoint)  # for backward compatibility

# Maps each symmetric joint type to its left/right AngleLandmark pair
_SYMMETRIC_PAIRS: dict[SymmetricJoint, tuple[AngleLandmark, AngleLandmark]] = {
    SymmetricJoint.shoulder: (AngleLandmark.left_shoulder, AngleLandmark.right_shoulder),
    SymmetricJoint.elbow: (AngleLandmark.left_elbow, AngleLandmark.right_elbow),
    SymmetricJoint.hip: (AngleLandmark.left_hip, AngleLandmark.right_hip),
    SymmetricJoint.knee: (AngleLandmark.left_knee, AngleLandmark.right_knee),
}

SYMMETRY_RANGE: tuple[float, float] = NORMALIZED_RANGE


class SymmetryFeature(NormalizedScalarFeature[SymmetricJoint]):
    """Symmetry scores for symmetric joint pairs (range [0, 1]).

    Measures how similar left/right joint angles are after mirroring.
    Values are symmetry scores [0, 1] per joint pair:
    - 1.0 = perfect symmetry (left and right angles identical after mirroring)
    - 0.0 = maximum asymmetry (left and right angles completely different)

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
    def feature_enum(cls) -> type[SymmetricJoint]:
        """Returns SymmetricJoint enum."""
        return SymmetricJoint

    # ========== SYMMETRY-SPECIFIC CONVENIENCE METHODS ==========

    def overall_symmetry(self, method: AggregationMethod = AggregationMethod.MEAN,
                        min_confidence: float = 0.0) -> float:
        """Compute overall symmetry score using specified aggregation method.

        Args:
            method: Statistical aggregation method (default: MEAN)
            min_confidence: Minimum confidence to include joint (default: 0.0)

        Returns:
            Overall symmetry score [0, 1], or NaN if no joints meet criteria

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
            >>> # For strict symmetric poses (all joints must be symmetric):
            >>> overall = symmetry.overall_symmetry(
            ...     AggregationMethod.HARMONIC_MEAN,
            ...     min_confidence=0.6
            ... )
        """
        return self.aggregate(method, min_confidence)


"""
=============================================================================
SYMMETRYFEATURE QUICK API REFERENCE
=============================================================================

Design Philosophy (from BaseFeature):
-------------------------------------
Raw Access (numpy-native):
  • feature.values      → Full array, shape (n_joints,) for symmetry scores
  • feature.scores      → Full confidence scores (n_joints,)
  • feature[joint]      → Single value (float, [0, 1])
  Use for: Numpy operations, batch processing, performance

Python-Friendly Access:
  • feature.get(joint, fill)    → Python float with NaN handling
  • feature.get_score(joint)    → Python float
  • feature.get_scores(joints)  → Python list
  Use for: Logic, conditionals, unpacking, defaults

Inherited from BaseScalarFeature (single value per joint):
----------------------------------------------------------
Properties:
  • values: np.ndarray                             All symmetry scores [0, 1]
  • scores: np.ndarray                             All confidence scores
  • valid_mask: np.ndarray                         Boolean validity mask
  • valid_count: int                               Number of valid symmetry scores
  • len(feature): int                              Total number of joint pairs (4)

Single Value Access:
  • feature[joint] -> float                        Get symmetry score [0, 1]
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
      Use for: When most joints should be symmetric

  • feature.harmonic_mean(min_confidence=0.0) -> float
      Confidence-weighted harmonic mean (heavily penalizes low values)
      Use for: When ALL joints must be symmetric (strict)

  • feature.aggregate(method, min_confidence=0.0) -> float
      General aggregation with method selection

  • feature.min_value(min_confidence=0.0) -> float
      Minimum symmetry score (least symmetric joint pair)

  • feature.max_value(min_confidence=0.0) -> float
      Maximum symmetry score (most symmetric joint pair)

  • feature.median(min_confidence=0.0) -> float
      Median symmetry score

  • feature.std(min_confidence=0.0) -> float
      Standard deviation of symmetry scores

SymmetryFeature-Specific Methods:
----------------------------------
  • feature.overall_symmetry(method=MEAN, min_confidence=0.0) -> float
      Convenience wrapper for aggregate() with symmetry-specific naming
      Recommended method names for symmetry:
      - AggregationMethod.MEAN: Balanced overall symmetry
      - AggregationMethod.GEOMETRIC_MEAN: Penalize asymmetric joints
      - AggregationMethod.HARMONIC_MEAN: Strict symmetry requirement

Common Usage Patterns:
----------------------
# Individual joint pair symmetry:
shoulder_sym = symmetry[SymmetricJoint.shoulder]
if shoulder_sym > 0.9:
    print("Shoulders are very symmetric!")

# Overall symmetry (mean - balanced):
overall = symmetry.overall_symmetry()
if overall > 0.85:
    print("Pose is highly symmetric!")

# Geometric mean (penalizes asymmetry):
overall = symmetry.overall_symmetry(AggregationMethod.GEOMETRIC_MEAN)
if overall > 0.80:
    print("Good overall symmetry (no major asymmetries)")

# Harmonic mean (strict - all must be symmetric):
overall = symmetry.overall_symmetry(AggregationMethod.HARMONIC_MEAN)
if overall > 0.75:
    print("Excellent symmetry (all joints symmetric!)")

# High-confidence only:
overall = symmetry.overall_symmetry(
    AggregationMethod.GEOMETRIC_MEAN,
    min_confidence=0.7
)

# Direct statistical methods:
mean_sym = symmetry.mean()
geom_sym = symmetry.geometric_mean()
harm_sym = symmetry.harmonic_mean()

# Check symmetry consistency:
std_sym = symmetry.std()
if std_sym < 0.1:
    print("Consistent symmetry across all joints!")

# Find least/most symmetric joints:
worst = symmetry.min_value()
best = symmetry.max_value()

# Process all joint pairs:
for joint in SymmetricJoint:
    if symmetry.get_valid(joint):
        score = symmetry[joint]
        conf = symmetry.get_score(joint)
        print(f"{joint.name}: {score:.2f} (conf: {conf:.2f})")

Statistical Comparison for Symmetry:
-------------------------------------
Given symmetry scores: [0.9, 0.9, 0.9, 0.3]
(3 symmetric joints, 1 asymmetric)

• Mean:          0.75  → "75% symmetric overall"
• Geometric:     0.69  → "69% symmetric (penalizes asymmetry)"
• Harmonic:      0.51  → "51% symmetric (very strict)"

Use case guidance:
- Mean:      General symmetry assessment (balanced)
- Geometric: Prefer symmetric poses, some tolerance
- Harmonic:  Require all joints symmetric (strict yoga/dance poses)

SymmetricJoint Enum Values:
---------------------------
  • shoulder (0)  - Left/right shoulder pair symmetry
  • elbow (1)     - Left/right elbow pair symmetry
  • hip (2)       - Left/right hip pair symmetry
  • knee (3)      - Left/right knee pair symmetry

Notes:
------
- Symmetry scores are in range [0.0, 1.0]
  * 1.0 = perfect symmetry (angles identical after mirroring)
  * 0.0 = maximum asymmetry (angles maximally different)
- Invalid scores are NaN (check with get_valid() before use)
- Confidence scores indicate reliability of the symmetry measurement
  (minimum confidence of the two angles being compared)
- Angles should already be mirrored before computing symmetry
  (right-side angles negated for symmetric representation)
- Zero symmetry (complete asymmetry) is replaced with TINY (1e-5) in
  geometric/harmonic means to preserve semantic meaning (penalizes score
  rather than being filtered out)
- All statistics support min_confidence filtering to ignore uncertain
  measurements
=============================================================================
"""