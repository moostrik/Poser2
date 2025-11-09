from enum import IntEnum

import numpy as np
from typing_extensions import Self

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
POINT_NUM_LANDMARKS: int = len(PointLandmark)
POINT_COORD_RANGE: tuple[float, float] = (0.0, 1.0)


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
    def joint_enum(cls) -> type[PointLandmark]:
        """Returns PoseJoint enum."""
        return PointLandmark

    @classmethod
    def dimensions(cls) -> int:
        """Returns 2 for 2D points (x, y)."""
        return 2

    @classmethod
    def default_range(cls) -> np.ndarray:
        """Returns normalized coordinate range [0.0, 1.0] for both x and y.

        Returns:
            Array of shape (2, 2): [[0.0, 1.0], [0.0, 1.0]]
                - Row 0: x range [0.0, 1.0]
                - Row 1: y range [0.0, 1.0]
        """
        # Use the constant to build the array
        min_val, max_val = POINT_COORD_RANGE
        return np.array([
            [min_val, max_val],  # x range
            [min_val, max_val],  # y range
        ], dtype=np.float32)

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
        n_joints = len(cls.joint_enum())
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
            >>> dist = feature1.distance_to(feature2, PoseJoint.nose)
        """
        p1 = self._values[joint]
        p2 = other._values[joint]

        # Return NaN if either point is invalid
        if np.any(np.isnan(p1)) or np.any(np.isnan(p2)):
            return np.nan

        return float(np.linalg.norm(p1 - p2))