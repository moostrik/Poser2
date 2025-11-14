"""
=============================================================================
ANGLEFEATURE API REFERENCE
=============================================================================

Concrete implementation of BaseScalarFeature for body landmark angles.

**IMPORTANT: Angles are normalized and mirrored for consistent pose comparison.**

Coordinate System Design:
-------------------------
AngleFeature uses a normalized coordinate system where:

1. **Normalization (Rotation Offsets)**:
   - All angles are rotated relative to a neutral standing position
   - Example: Elbow at 0° = neutral position (slightly bent), not raw geometric angle
   - Applied via rotation offsets per landmark (see AngleFactory._ANGLE_OFFSET)
   - Each landmark has a specific offset to define its "neutral" state

2. **Mirroring (Right-Side Negation)**:
   - Right-side angles are negated for symmetric representation
   - Enables direct left/right comparison (e.g., left_elbow vs right_elbow)
   - Applied to: right_shoulder, right_elbow, right_hip, right_knee

This coordinate system is established at construction (AngleFactory.from_points())
and enables:
  • Direct symmetry comparison (SymmetryFeature)
  • Symmetric pose visualization
  • Pose matching across mirror poses
  • Intuitive angle interpretation (0° = neutral position)

Use for: pose articulation analysis, angle tracking, pose comparison, symmetry assessment.

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
  • AngleFeature(values, scores)           → Direct (fast, no validation)
  • AngleFeature.create_empty()            → All NaN values, zero scores

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
  • A scalar angle value (float) in radians [-π, π] - may be NaN for invalid/missing data
  • A confidence score [0.0, 1.0]

Storage:
  • values: np.ndarray, shape (n_elements,), dtype float32, range [-π, π]
  • scores: np.ndarray, shape (n_elements,), dtype float32

Properties:
-----------
  • values: np.ndarray                             All angle values (n_elements,)
  • scores: np.ndarray                             All confidence scores (n_elements,)
  • valid_mask: np.ndarray                         Boolean validity mask (n_elements,)
  • valid_count: int                               Number of valid angles
  • len(feature): int                              Total number of elements (9)

Single Value Access:
--------------------
  • feature[element] -> float                      Get angle in radians (supports enum or int)
  • feature.get(element, fill=np.nan) -> float     Get angle with NaN handling
  • feature.get_value(element, fill) -> float      Alias for get()
  • feature.get_score(element) -> float            Get confidence score
  • feature.get_valid(element) -> bool             Check if angle is valid

Batch Operations:
-----------------
  • feature.get_values(elements, fill) -> list[float]  Get multiple angles
  • feature.get_scores(elements) -> list[float]        Get multiple scores
  • feature.are_valid(elements) -> bool                Check if ALL valid

Factory Methods:
----------------
  • AngleFeature.create_empty() -> AngleFeature          All NaN angles, zero scores

Validation:
-----------
  • feature.validate(check_ranges=True) -> tuple[bool, str|None]
      Returns (is_valid, error_message)

Implemented Methods (from BaseScalarFeature):
----------------------------------------------
  • feature_enum() -> type[AngleLandmark]          Returns AngleLandmark enum
  • default_range() -> tuple[float, float]         Returns SYMMETRIC_PI_RANGE (-π, π)

AngleFeature-Specific:
======================

Angle Math:
-----------
  • feature.subtract(other) -> AngleFeature
      Compute angular differences with proper wrapping
      Returns shortest angular distance between corresponding angles
      Confidence is minimum of the two source confidences (conservative)

Degree Conversion:
------------------
  • feature.get_degree(element, fill=np.nan) -> float
      Get angle in degrees (optionally replacing NaN with fill value)

  • feature.to_degrees() -> np.ndarray
      Convert all angles to degrees (returns numpy array)

AngleLandmark Enum:
-------------------
  • left_shoulder (0)    - Shoulder angle
  • right_shoulder (1)   - Shoulder angle
  • left_elbow (2)       - Elbow angle
  • right_elbow (3)      - Elbow angle
  • left_hip (4)         - Hip angle
  • right_hip (5)        - Hip angle
  • left_knee (6)        - Knee angle
  • right_knee (7)       - Knee angle
  • head (8)             - Head yaw angle (special calculation)

Notes:
------
- Angles are in radians [-π, π] (±180 degrees)
- Right-side angles are mirrored for symmetric representation
- Head yaw is computed from eye/shoulder positions (special case)
- Invalid/uncomputable angles are NaN with score 0.0
- Use get_degree() for degree conversion (more convenient than manual conversion)
- Use subtract() instead of direct subtraction to handle angle wrapping and scores
- Arrays are read-only after construction (immutable)
- Use validate() for debugging, not in production loops
- Constructor takes ownership - caller must not modify arrays after passing
=============================================================================
"""
from enum import IntEnum

import numpy as np

from modules.pose.features.base.BaseFeature import SYMMETRIC_PI_RANGE
from modules.pose.features.base.BaseScalarFeature import BaseScalarFeature


class AngleLandmark(IntEnum):
    """Body landmark angles for pose articulation measurement.

    Angles are measured in radians [-π, π]:
    - 0 = neutral/straight position
    - Positive = one direction of rotation
    - Negative = opposite direction of rotation
    """
    left_shoulder = 0
    right_shoulder = 1
    left_elbow = 2
    right_elbow = 3
    left_hip = 4
    right_hip = 5
    left_knee = 6
    right_knee = 7
    head = 8  # Head yaw (special calculation)


# Constants
ANGLE_LANDMARK_NAMES: list[str] = [e.name for e in AngleLandmark]
ANGLE_NUM_LANDMARKS: int = len(AngleLandmark) # for backward compatibility
ANGLE_RANGE: tuple[float, float] = SYMMETRIC_PI_RANGE # for backward compatibility


class Angles(BaseScalarFeature[AngleLandmark]):
    """Angles for body landmarks (radians, range [-π, π]).

    Represents angles at articulation points for pose analysis:
    - Each landmark has a single angle value in radians
    - Angles are in range [-π, π] (±180 degrees)
    - Invalid/uncomputable angles are NaN
    - Each angle has a confidence score [0.0, 1.0]
    """

    # ========== ABSTRACT METHOD IMPLEMENTATIONS ==========

    @classmethod
    def feature_enum(cls) -> type[AngleLandmark]:
        """Returns AngleLandmark enum."""
        return AngleLandmark

    @classmethod
    def default_range(cls) -> tuple[float, float]:
        """Returns angle range in radians [-π, π]."""
        return ANGLE_RANGE

    # ========== RAW ANGLE-SPECIFIC OPERATIONS =========

    def subtract(self, other: 'Angles') -> 'Angles':
        """Compute angular differences with proper wrapping (batch operation).

        Calculates the shortest angular distance between corresponding angles
        in two AngleFeature instances.

        The confidence score for each delta is the minimum of the two source
        confidences (conservative approach).
        """
        diff: np.ndarray = self.values - other.values
        # Wrap angles to [-π, π] range (shortest angular distance)
        wrapped_diff: np.ndarray = np.arctan2(np.sin(diff), np.cos(diff))
        min_scores: np.ndarray = np.minimum(self.scores, other.scores)
        return Angles(values=wrapped_diff, scores=min_scores)

    # ========== CONVENIENCE ACCESSORS ==========

    def get_degree(self, element: AngleLandmark | int, fill: float = np.nan) -> float:
        """Get angle in degrees (optionally replacing NaN with fill value)."""
        angle_rad = self._values[element]

        if np.isnan(angle_rad):
            return fill

        return float(np.degrees(angle_rad))

    def to_degrees(self) -> np.ndarray:
        """Convert all angles to degrees."""
        return np.degrees(self._values)
