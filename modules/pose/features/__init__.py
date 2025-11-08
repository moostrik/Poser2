"""Pose feature extraction and analysis.

This module provides classes for computing and analyzing pose features:
- PosePoints: Raw keypoint coordinates and scores
- PoseAngles: Joint angles computed from keypoints
- PoseAngleSimilarity: Similarity scores between pose pairs
- PoseAngleSymmetry: Symmetry analysis of poses
"""

from enum import Enum
from typing import Union

# Import all feature classes
from modules.pose.features.PosePoints import (
    PosePointData,
    PoseJoint,
    POSE_JOINT_NAMES,
    POSE_NUM_JOINTS,
    POSE_JOINT_COLORS,
    POSE_POINTS_RANGE
)

from modules.pose.features.PoseAngles import (
    PoseAngleData,
    AngleJoint,
    PoseAngleFactory,
    ANGLE_JOINT_NAMES,
    ANGLE_NUM_JOINTS,
    ANGLE_RANGE
)

from modules.pose.features.PoseAngleSymmetry import (
    PoseAngleSymmetryData,
    SymmetricJoint,
    PoseAngleSymmetryFactory,
    SYMM_JOINT_NAMES,
    SYMM_NUM_JOINTS,
    POSE_SYMMETRY_RANGE
)

# Feature type enum for dynamic dispatch
class PoseFeatureType(Enum):
    """Enum for different pose feature types."""
    POINTS = "points"
    ANGLES = "angles"
    DELTA = "delta"
    SYMMETRY = "symmetry"


# Type alias for any feature data
PoseFeatureData = Union[
    PosePointData,
    PoseAngleData,
    PoseAngleSymmetryData,
]

POSE_FEATURE_CLASSES: dict[PoseFeatureType, type] = {
    PoseFeatureType.POINTS: PosePointData,
    PoseFeatureType.ANGLES: PoseAngleData,
    PoseFeatureType.DELTA: PoseAngleData,
    PoseFeatureType.SYMMETRY: PoseAngleSymmetryData,
}

POSE_FEATURE_RANGES: dict[PoseFeatureType, tuple[float, float]] = {
    PoseFeatureType.POINTS: POSE_POINTS_RANGE,
    PoseFeatureType.ANGLES: ANGLE_RANGE,
    PoseFeatureType.DELTA: ANGLE_RANGE,
    PoseFeatureType.SYMMETRY: POSE_SYMMETRY_RANGE,
}

POSE_FEATURE_DIMENSIONS: dict[PoseFeatureType, int] = {
    PoseFeatureType.POINTS: 2,      # (x, y) coordinates
    PoseFeatureType.ANGLES: 1,      # Single angle value
    PoseFeatureType.DELTA: 1,       # Single delta angle value
    PoseFeatureType.SYMMETRY: 1,    # Single symmetry value
}

POSE_CLASS_TO_FEATURE_TYPE: dict[type, PoseFeatureType] = {
    PosePointData: PoseFeatureType.POINTS,
    PoseAngleData: PoseFeatureType.ANGLES,
    PoseAngleSymmetryData: PoseFeatureType.SYMMETRY,
}