"""
=============================================================================
POINT2DFEATURE API REFERENCE
=============================================================================

Concrete implementation of BaseVectorFeature for 2D body landmark positions.

Use for: pose keypoint tracking, position analysis, spatial relationships.

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
  • Point2DFeature(values, scores)           → Direct (fast, no validation)
  • Point2DFeature.create_empty()            → All NaN values, zero scores

Validation:
  • Asserts in constructors (removed with -O flag for production)
  • validate() method for debugging/testing/untrusted input
  • Fast by default, validate only when needed

Performance:
  Fast:     Property access, indexing, cached properties, array ops
  Moderate: get(), get_score() (Python conversion)
  Slow:     get_values(), get_scores() (iteration), validate()

Inherited from BaseVectorFeature:
==================================

Structure:
----------
Each element has:
  • A 2D vector (x, y coordinates) in [0.0, 1.0] range - may be NaN for invalid/missing data
  • A confidence score [0.0, 1.0]
  • A vector is INVALID if ANY component (x or y) is NaN

Storage:
  • values: np.ndarray, shape (n_elements, 2), dtype float32, range [0.0, 1.0]
  • scores: np.ndarray, shape (n_elements,), dtype float32

Properties:
-----------
  • values: np.ndarray                             All vectors (n_elements, 2)
  • scores: np.ndarray                             All confidence scores (n_elements,)
  • valid_mask: np.ndarray                         Boolean validity mask (n_elements,)
  • valid_count: int                               Number of valid vectors
  • len(feature): int                              Total number of elements (17)

Single Vector Access:
---------------------
  • feature[element] -> np.ndarray                 Get 2D point [x, y] (supports enum or int)
                                                   Returns (2,) array, may contain NaN
  • feature.get_score(element) -> float            Get confidence score
  • feature.get_valid(element) -> bool             Check if vector is valid

Batch Operations:
-----------------
  • feature.get_scores(elements) -> list[float]    Get multiple scores
  • feature.are_valid(elements) -> bool            Check if ALL valid

Factory Methods:
----------------
  • Point2DFeature.create_empty() -> Point2DFeature          All NaN vectors, zero scores

Validation:
-----------
  • feature.validate(check_ranges=True) -> tuple[bool, str|None]
      Returns (is_valid, error_message)

Implemented Methods (from BaseVectorFeature):
----------------------------------------------
  • feature_enum() -> type[PointLandmark]          Returns PointLandmark enum
  • dimensions() -> int                            Returns 2 for 2D points
  • default_range() -> tuple[float, float]         Returns NORMALIZED_RANGE (0.0, 1.0)

Point2DFeature-Specific:
========================

Coordinate Access:
------------------
  • feature.get_x(element: PointLandmark | int, fill: float = np.nan) -> float
      Get x coordinate (optionally replacing NaN with fill value)

  • feature.get_y(element: PointLandmark | int, fill: float = np.nan) -> float
      Get y coordinate (optionally replacing NaN with fill value)

  • feature.get(element: PointLandmark | int, fill: float = np.nan) -> tuple[float, float]
      Get (x, y) tuple (optionally replacing NaN with fill value)

Factory Methods (specialized):
-------------------------------
  • Point2DFeature.from_xy_arrays(x, y, scores?) -> Point2DFeature
      Create from separate x and y coordinate arrays

  • Point2DFeature.from_flat_array(flat, scores?) -> Point2DFeature
      Create from flat array [x0, y0, x1, y1, x2, y2, ...]
      Useful for MediaPipe format conversion

Array Conversions:
------------------
  • feature.to_flat_array() -> np.ndarray
      Convert to flat array [x0, y0, x1, y1, x2, y2, ...]
      Returns array of shape (n_elements * 2,)

  • feature.get_xy_arrays() -> tuple[np.ndarray, np.ndarray]
      Get separate x and y coordinate arrays
      Returns (x_array, y_array), each of shape (n_elements,)

Utilities:
----------
  • feature.distance_to(other, element) -> float
      Calculate Euclidean distance between element positions in two features
      Returns NaN if either point is invalid

PointLandmark Enum:
-------------------
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
- A point is INVALID if ANY component (x or y) is NaN (entire vector marked invalid)
- Invalid vectors must have score 0.0
- default_range() applies to BOTH dimensions (x and y use same range)
- Arrays are read-only after construction (immutable)
- Use validate() for debugging, not in production loops
- Constructor takes ownership - caller must not modify arrays after passing
=============================================================================
"""

from enum import IntEnum

import numpy as np
from typing_extensions import Self

from modules.pose.features.base.BaseFeature import NORMALIZED_RANGE
from modules.pose.features.base.BaseVectorFeature import BaseVectorFeature


class PointLandmark(IntEnum):
    """Enumeration of body landmarks for pose estimation."""
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
    """2D point coordinates for body landmarks (normalized [0, 1] range).

    Represents 2D keypoint positions for pose estimation, where:
    - Each landmark has (x, y) coordinates
    - Coordinates are normalized to [0.0, 1.0] range
    - Invalid/undetected landmarks have NaN coordinates
    - Each landmark has a confidence score [0.0, 1.0]
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

    def get_x(self, element: PointLandmark | int, fill: float = np.nan) -> float:
        """Get x coordinate for an element (optionally replacing NaN with fill value)."""
        x = float(self._values[element, 0])
        if not np.isnan(fill) and np.isnan(x):
            return fill
        return x

    def get_y(self, element: PointLandmark | int, fill: float = np.nan) -> float:
        """Get y coordinate for an element (optionally replacing NaN with fill value)."""
        y = float(self._values[element, 1])
        if not np.isnan(fill) and np.isnan(y):
            return fill
        return y

    def get(self, element: PointLandmark | int, fill: float = np.nan) -> tuple[float, float]:
        """Get (x, y) tuple for an element (optionally replacing NaN with fill value)."""
        x = self.get_x(element, fill=fill)
        y = self.get_y(element, fill=fill)
        return (x, y)

    # ========== SPECIALIZED CONSTRUCTORS ==========

    @classmethod
    def from_xy_arrays(cls, x: np.ndarray, y: np.ndarray, scores: np.ndarray) -> Self:
        """Create from separate x and y coordinate arrays.

        Args:
            x: X coordinates (length n_landmarks)
            y: Y coordinates (length n_landmarks)
            scores: Optional confidence scores. If None, generates from validity.

        Returns:
            New Point2DFeature instance

        Examples:
            >>> x = np.array([0.5, 0.3, 0.7, ...])  # 17 values
            >>> y = np.array([0.6, 0.4, 0.8, ...])  # 17 values
            >>> feature = Point2DFeature.from_xy_arrays(x, y)
        """
        # Stack into (n_elements, 2) shape
        values = np.column_stack([x, y]).astype(np.float32)
        return cls(values, scores)

    @classmethod
    def from_flat_array(cls, flat: np.ndarray, scores: np.ndarray) -> Self:
        """Create from flat array [x0, y0, x1, y1, x2, y2, ...].

        Args:
            flat: Flat array of alternating x, y values (length n_elements * 2)
            scores: Optional confidence scores. If None, generates from validity.

        Returns:
            New Point2DFeature instance

        Examples:
            >>> # MediaPipe format: [x0, y0, x1, y1, ...]
            >>> flat = np.array([0.5, 0.6, 0.3, 0.4, ...])  # 34 values (17*2)
            >>> feature = Point2DFeature.from_flat_array(flat)
        """
        # Reshape to (n_elements, 2)
        n_elements = len(cls.feature_enum())
        values = flat.reshape(n_elements, 2).astype(np.float32)
        return cls(values, scores)

    # ========== UTILITY METHODS ==========

    def to_flat_array(self) -> np.ndarray:
        """Convert to flat array [x0, y0, x1, y1, x2, y2, ...].

        Returns:
            Flat array of shape (n_elements * 2,)

        Examples:
            >>> feature = Point2DFeature(...)
            >>> flat = feature.to_flat_array()  # Shape: (34,)
        """
        return self._values.flatten()

    def get_xy_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Get separate x and y coordinate arrays.

        Returns:
            Tuple of (x_array, y_array), each of shape (n_elements,)

        Examples:
            >>> x, y = feature.get_xy_arrays()
            >>> x.shape  # (17,)
            >>> y.shape  # (17,)
        """
        return self._values[:, 0], self._values[:, 1]

    def distance_to(self, other: 'Point2DFeature', element: PointLandmark | int) -> float:
        """Calculate Euclidean distance between element positions in two features.

        Args:
            other: Another Point2DFeature to compare against
            element: landmark to measure distance for

        Returns:
            Euclidean distance, or NaN if either point is invalid

        Examples:
            >>> dist = feature1.distance_to(feature2, PointLandmark.nose)
        """
        p1 = self._values[element]
        p2 = other._values[element]

        # Return NaN if either point is invalid
        if np.any(np.isnan(p1)) or np.any(np.isnan(p2)):
            return np.nan

        return float(np.linalg.norm(p1 - p2))

