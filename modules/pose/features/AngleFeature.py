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


class AngleFeature(BaseScalarFeature[AngleLandmark]):
    """Joint angles for body landmarks (radians, range [-π, π]).

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

    def subtract(self, other: 'AngleFeature') -> 'AngleFeature':
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
        return AngleFeature(values=wrapped_diff, scores=min_scores)

    # ========== CONVENIENCE ACCESSORS ==========

    def get_degree(self, landmark: AngleLandmark | int, fill: float = np.nan) -> float:
        """Get angle in degrees (optionally replacing NaN with fill value)."""
        angle_rad = self._values[landmark]

        if np.isnan(angle_rad):
            return fill

        return float(np.degrees(angle_rad))

    def to_degrees(self) -> np.ndarray:
        """Convert all angles to degrees."""
        return np.degrees(self._values)



"""
=============================================================================
ANGLEFEATURE QUICK API REFERENCE
=============================================================================

Design Philosophy (from BaseFeature):
-------------------------------------
Raw Access (numpy-native):
  • feature.values      → Full array, shape (n_joints,) for angles
  • feature.scores      → Full scores (n_joints,)
  • feature[joint]      → Single value (float, radians) - supports enum or int
  Use for: Numpy operations, batch processing, performance

Python-Friendly Access:
  • feature.get(joint, fill)    → Python float with NaN handling
  • feature.get_score(joint)    → Python float
  • feature.get_scores(joints)  → Python list
  Use for: Logic, conditionals, unpacking, defaults

Inherited from BaseScalarFeature (single value per joint):
----------------------------------------------------------
Properties:
  • values: np.ndarray                             All angle values (radians)
  • scores: np.ndarray                             All confidence scores
  • valid_mask: np.ndarray                         Boolean validity mask
  • valid_count: int                               Number of valid angles
  • len(feature): int                              Total number of joints (9)

Single Value Access:
  • feature[joint] -> float                        Get angle in radians (supports enum or int)
  • feature.get(joint, fill=0.0) -> float          Get angle with NaN fill
  • feature.get_value(joint, fill) -> float        Alias for get()
  • feature.get_score(joint) -> float              Get confidence score
  • feature.get_valid(joint) -> bool               Check if angle is valid

Batch Operations:
  • feature.get_values(joints, fill) -> list[float]  Get multiple angles
  • feature.get_scores(joints) -> list[float]        Get multiple scores
  • feature.are_valid(joints) -> bool                Check if ALL valid

Factory Methods:
  • AngleFeature.create_empty() -> AngleFeature      All NaN angles
  • AngleFeature.from_values(values, scores)         Create with validation
  • AngleFeature.create_validated(values, scores)    Create with strict checks

AngleFeature-Specific Methods:
-------------------------------
Degree Conversion:
  • feature.get_degree(joint, fill=np.nan) -> float  Get angle in degrees
  • feature.to_degrees() -> np.ndarray               All angles in degrees

Angle Math:
  • feature.subtract(other) -> AngleFeature          Angular difference with wrapping [-π, π]
      Computes shortest angular distance between corresponding angles in two poses.
      Confidence is minimum of the two source confidences (conservative).

Common Usage Patterns:
----------------------
# Get angle in degrees:
angle_deg = angles.get_degree(AngleLandmark.left_elbow)

# Direct access via bracket notation:
angle_rad = angles[AngleLandmark.left_elbow]  # Returns float in radians

# Check if angle computation was successful:
if angles.get_valid(AngleLandmark.left_knee):
    angle = angles[AngleLandmark.left_knee]
    confidence = angles.get_score(AngleLandmark.left_knee)

# Process only valid angles:
for joint in AngleLandmark:
    if angles.get_valid(joint):
        print(f"{joint.name}: {angles.get_degree(joint):.1f}°")

# Compare angles between two poses (proper angular wrapping):
angle_delta = pose1_angles.subtract(pose2_angles)
left_elbow_diff = angle_delta[AngleLandmark.left_elbow]

# Convert all to degrees for display:
all_degrees = angles.to_degrees()

# Batch processing (numpy-native):
valid_angles = angles.values[angles.valid_mask]  # Only valid angles
all_scores = angles.scores  # Raw numpy array

# Batch validation and extraction:
joints = [AngleLandmark.left_elbow, AngleLandmark.left_knee, AngleLandmark.left_hip]
if angles.are_valid(joints):
    values = angles.get_values(joints)
    scores = angles.get_scores(joints)

AngleLandmark Enum Values:
--------------------------
  • left_shoulder (0)    - Shoulder joint angle
  • right_shoulder (1)   - Shoulder joint angle
  • left_elbow (2)       - Elbow joint angle
  • right_elbow (3)      - Elbow joint angle
  • left_hip (4)         - Hip joint angle
  • right_hip (5)        - Hip joint angle
  • left_knee (6)        - Knee joint angle
  • right_knee (7)       - Knee joint angle
  • head (8)             - Head yaw angle

Notes:
------
- Angles are in radians [-π, π] (use get_degree() for degrees)
- Right-side angles are mirrored for symmetric representation
- Head yaw is computed from eye/shoulder positions (special case)
- Invalid angles are NaN (check with get_valid() before use)
- Confidence scores indicate reliability of angle computation
- Use subtract() instead of direct subtraction to handle angle wrapping
=============================================================================
"""


