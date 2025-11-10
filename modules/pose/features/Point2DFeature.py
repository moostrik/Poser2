from enum import IntEnum

import numpy as np
from typing_extensions import Self

from modules.pose.features.base.BaseFeature import NORMALIZED_RANGE
from modules.pose.features.base.BaseVectorFeature import BaseVectorFeature


class PointLandmark(IntEnum):
    """Enumeration of body joints for pose estimation."""
    nose =          0
    left_eye =      1
    right_eye =     2
    left_ear =      3
    right_ear =     4
    left_shoulder = 5
    right_shoulder= 6
    left_elbow =    7
    right_elbow =   8
    left_wrist =    9
    right_wrist =   10
    left_hip =      11
    right_hip =     12
    left_knee =     13
    right_knee =    14
    left_ankle =    15
    right_ankle =   16


# Constants
POINT_LANDMARK_NAMES: list[str] = [e.name for e in PointLandmark]
POINT_NUM_LANDMARKS: int = len(PointLandmark) # for backward compatibility
POINT2D_COORD_RANGE: tuple[float, float] = NORMALIZED_RANGE # for backward compatibility


class Point2DFeature(BaseVectorFeature[PointLandmark]):
    """2D point coordinates for body joints (normalized [0, 1] range).

    Represents 2D keypoint positions for pose estimation, where:
    - Each joint has (x, y) coordinates
    - Coordinates are normalized to [0.0, 1.0] range
    - Invalid/undetected joints have NaN coordinates
    - Each joint has a confidence score [0.0, 1.0]
    """

    # ========== ABSTRACT METHOD IMPLEMENTATIONS ==========

    @classmethod
    def feature_enum(cls) -> type[PointLandmark]:
        """Returns PointLandmark enum."""
        return PointLandmark

    @classmethod
    def dimensions(cls) -> int:
        """Returns 2 for 2D points (x, y)."""
        return 2

    @classmethod
    def default_range(cls) -> tuple[float, float]:
        """Returns normalized coordinate range"""
        return NORMALIZED_RANGE

    # ========== CONVENIENCE ACCESSORS ==========

    def get_x(self, joint: PointLandmark | int, fill: float = np.nan) -> float:
        """Get x coordinate (optionally replacing NaN with fill value)."""
        x = float(self._values[joint, 0])
        if not np.isnan(fill) and np.isnan(x):
            return fill
        return x

    def get_y(self, joint: PointLandmark | int, fill: float = np.nan) -> float:
        """Get y coordinate (optionally replacing NaN with fill value)."""
        y = float(self._values[joint, 1])
        if not np.isnan(fill) and np.isnan(y):
            return fill
        return y

    def get(self, joint: PointLandmark | int, fill: float = np.nan) -> tuple[float, float]:
        """Get (x, y) tuple (optionally replacing NaN with fill value)."""
        x = self.get_x(joint, fill=fill)
        y = self.get_y(joint, fill=fill)
        return (x, y)



    # ========== SPECIALIZED CONSTRUCTORS ==========

    @classmethod
    def from_xy_arrays(cls, x: np.ndarray, y: np.ndarray, scores: np.ndarray | None = None) -> Self:
        """Create from separate x and y coordinate arrays.

        Args:
            x: X coordinates (length n_joints)
            y: Y coordinates (length n_joints)
            scores: Optional confidence scores. If None, generates from validity.

        Returns:
            New Point2DFeature instance

        Examples:
            >>> x = np.array([0.5, 0.3, 0.7, ...])  # 17 values
            >>> y = np.array([0.6, 0.4, 0.8, ...])  # 17 values
            >>> feature = Point2DFeature.from_xy_arrays(x, y)
        """
        # Stack into (n_joints, 2) shape
        values = np.column_stack([x, y]).astype(np.float32)
        return cls.from_values(values, scores)

    @classmethod
    def from_flat_array(cls, flat: np.ndarray, scores: np.ndarray | None = None) -> Self:
        """Create from flat array [x0, y0, x1, y1, x2, y2, ...].

        Args:
            flat: Flat array of alternating x, y values (length n_joints * 2)
            scores: Optional confidence scores. If None, generates from validity.

        Returns:
            New Point2DFeature instance

        Examples:
            >>> # MediaPipe format: [x0, y0, x1, y1, ...]
            >>> flat = np.array([0.5, 0.6, 0.3, 0.4, ...])  # 34 values (17*2)
            >>> feature = Point2DFeature.from_flat_array(flat)
        """
        # Reshape to (n_joints, 2)
        n_joints = len(cls.feature_enum())
        values = flat.reshape(n_joints, 2).astype(np.float32)
        return cls.from_values(values, scores)

    # ========== UTILITY METHODS ==========

    def to_flat_array(self) -> np.ndarray:
        """Convert to flat array [x0, y0, x1, y1, x2, y2, ...].

        Returns:
            Flat array of shape (n_joints * 2,)

        Examples:
            >>> feature = Point2DFeature(...)
            >>> flat = feature.to_flat_array()  # Shape: (34,)
        """
        return self._values.flatten()

    def get_xy_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Get separate x and y coordinate arrays.

        Returns:
            Tuple of (x_array, y_array), each of shape (n_joints,)

        Examples:
            >>> x, y = feature.get_xy_arrays()
            >>> x.shape  # (17,)
            >>> y.shape  # (17,)
        """
        return self._values[:, 0], self._values[:, 1]

    def distance_to(self, other: 'Point2DFeature', joint: PointLandmark | int) -> float:
        """Calculate Euclidean distance between joint positions in two features.

        Args:
            other: Another Point2DFeature to compare against
            joint: Joint to measure distance for

        Returns:
            Euclidean distance, or NaN if either point is invalid

        Examples:
            >>> dist = feature1.distance_to(feature2, PointLandmark.nose)
        """
        p1 = self._values[joint]
        p2 = other._values[joint]

        # Return NaN if either point is invalid
        if np.any(np.isnan(p1)) or np.any(np.isnan(p2)):
            return np.nan

        return float(np.linalg.norm(p1 - p2))


"""
=============================================================================
POINT2DFEATURE QUICK API REFERENCE
=============================================================================

Design Philosophy (from BaseFeature):
-------------------------------------
Raw Access (numpy-native):
  • feature.values      → Full array, shape (n_joints, 2) for 2D points
  • feature.scores      → Full scores (n_joints,)
  • feature[joint]      → Single vector (np.ndarray with shape (2,))
  Use for: Numpy operations, batch processing, performance

Python-Friendly Access:
  • feature.get(joint, fill)    → Python tuple (x, y) with NaN handling
  • feature.get_x(joint, fill)  → Python float (x coordinate)
  • feature.get_y(joint, fill)  → Python float (y coordinate)
  • feature.get_score(joint)    → Python float
  • feature.get_scores(joints)  → Python list
  Use for: Logic, conditionals, unpacking, defaults

Inherited from BaseVectorFeature (multi-dimensional values per joint):
----------------------------------------------------------------------
Properties:
  • values: np.ndarray                             All point coordinates (n_joints, 2)
  • scores: np.ndarray                             All confidence scores (n_joints,)
  • valid_mask: np.ndarray                         Boolean validity mask (n_joints,)
  • valid_count: int                               Number of valid points
  • len(feature): int                              Total number of joints (17)

Single Value Access:
  • feature[joint] -> np.ndarray                   Get point as array [x, y]
  • feature.get(joint, fill) -> tuple[float, float] Get point as (x, y) tuple
  • feature.get_x(joint, fill) -> float            Get x coordinate only
  • feature.get_y(joint, fill) -> float            Get y coordinate only
  • feature.get_score(joint) -> float              Get confidence score
  • feature.get_valid(joint) -> bool               Check if point is valid

Batch Operations:
  • feature.get_scores(joints) -> list[float]      Get multiple scores
  • feature.are_valid(joints) -> bool              Check if ALL valid

Factory Methods:
  • Point2DFeature.create_empty() -> Point2DFeature           All NaN coordinates
  • Point2DFeature.from_values(values, scores)                Create with validation
  • Point2DFeature.from_xy_arrays(x, y, scores)               Create from separate x, y arrays
  • Point2DFeature.from_flat_array(flat, scores)              Create from [x0,y0,x1,y1,...] format
  • Point2DFeature.create_validated(values, scores)           Create with strict checks

Point2DFeature-Specific Methods:
--------------------------------
Coordinate Access:
  • feature.get_x(joint, fill=np.nan) -> float     Get x coordinate with fill
  • feature.get_y(joint, fill=np.nan) -> float     Get y coordinate with fill
  • feature.get(joint, fill=np.nan) -> tuple       Get (x, y) tuple with fill

Array Conversions:
  • feature.to_flat_array() -> np.ndarray          Convert to [x0,y0,x1,y1,...] format
  • feature.get_xy_arrays() -> tuple[np.ndarray, np.ndarray]  Get separate x, y arrays

Utilities:
  • feature.distance_to(other, joint) -> float     Euclidean distance to another feature

Common Usage Patterns:
----------------------
# Get point coordinates:
x, y = points.get(PointLandmark.nose)
x_only = points.get_x(PointLandmark.nose)

# Check if point is valid before using:
if points.get_valid(PointLandmark.nose):
    coords = points[PointLandmark.nose]  # np.ndarray [x, y]
    confidence = points.get_score(PointLandmark.nose)

# Process only valid points:
for joint in PointLandmark:
    if points.get_valid(joint):
        x, y = points.get(joint)
        print(f"{joint.name}: ({x:.3f}, {y:.3f})")

# Check if multiple points are valid (e.g., for angle calculation):
keypoints = [PointLandmark.left_shoulder, PointLandmark.left_elbow, PointLandmark.left_wrist]
if points.are_valid(keypoints):
    # All points valid - safe to compute angle
    p1, p2, p3 = [points[kp] for kp in keypoints]

# Batch processing (numpy-native):
valid_points = points.values[points.valid_mask]  # Only valid coordinates
all_x = points.values[:, 0]  # All x coordinates
all_y = points.values[:, 1]  # All y coordinates

# Calculate distance between poses:
distance = pose1.distance_to(pose2, PointLandmark.nose)

# Convert formats:
flat = points.to_flat_array()  # [x0, y0, x1, y1, ...]
x_array, y_array = points.get_xy_arrays()

# Create from different formats:
points = Point2DFeature.from_xy_arrays(x_coords, y_coords)
points = Point2DFeature.from_flat_array(mediapipe_output)

PointLandmark Enum Values:
--------------------------
  • nose (0)           - Nose tip
  • left_eye (1)       - Left eye center
  • right_eye (2)      - Right eye center
  • left_ear (3)       - Left ear
  • right_ear (4)      - Right ear
  • left_shoulder (5)  - Left shoulder
  • right_shoulder (6) - Right shoulder
  • left_elbow (7)     - Left elbow
  • right_elbow (8)    - Right elbow
  • left_wrist (9)     - Left wrist
  • right_wrist (10)   - Right wrist
  • left_hip (11)      - Left hip
  • right_hip (12)     - Right hip
  • left_knee (13)     - Left knee
  • right_knee (14)    - Right knee
  • left_ankle (15)    - Left ankle
  • right_ankle (16)   - Right ankle

Notes:
------
- Coordinates are normalized to [0.0, 1.0] range
- Invalid/undetected joints have NaN coordinates
- A point is invalid if ANY component (x or y) is NaN
- Confidence scores indicate detection reliability [0.0, 1.0]
- Use get() methods for Python-friendly access with fill values
- Use direct indexing feature[joint] for numpy operations
=============================================================================
"""