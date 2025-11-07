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
    ANGLE_JOINT_NAMES,
    ANGLE_NUM_JOINTS,
    ANGLE_RANGE
)

from modules.pose.features.PoseAngleSimilarity import (
    PoseAngleSimilarityData,
    PoseSimilarityBatch,
    PoseSimilarityBatchCallback,
    POSE_SIMILARITY_RANGE
)

from modules.pose.features.PoseAngleSymmetry import (
    PoseAngleSymmetryData,
    POSE_SYMMETRY_RANGE
)

from modules.pose.features.PoseAngleFeatureBase import (
    PoseAngleFeatureBase,
    FeatureStatistic,
)


# Feature type enum for dynamic dispatch
class PoseFeatureType(Enum):
    """Enum for different pose feature types."""
    POINTS = "points"
    ANGLES = "angles"
    SIMILARITY = "similarity"
    SYMMETRY = "symmetry"


# Type alias for any feature data
PoseFeatureData = Union[
    PosePointData,
    PoseAngleData,
    PoseAngleSimilarityData,
    PoseAngleSymmetryData,
]

POSE_FEATURE_CLASSES: dict[PoseFeatureType, type] = {
    PoseFeatureType.POINTS: PosePointData,
    PoseFeatureType.ANGLES: PoseAngleData,
    PoseFeatureType.SIMILARITY: PoseAngleSimilarityData,
    PoseFeatureType.SYMMETRY: PoseAngleSymmetryData,
}

POSE_FEATURE_RANGES: dict[PoseFeatureType, tuple[float, float] | None] = {
    PoseFeatureType.POINTS: POSE_POINTS_RANGE,
    PoseFeatureType.ANGLES: ANGLE_RANGE,
    PoseFeatureType.SIMILARITY: POSE_SIMILARITY_RANGE,
    PoseFeatureType.SYMMETRY: POSE_SYMMETRY_RANGE,
}

POSE_FEATURE_DIMENSIONS: dict[PoseFeatureType, int] = {
    PoseFeatureType.POINTS: 2,      # (x, y) coordinates
    PoseFeatureType.ANGLES: 1,      # Single angle value
    PoseFeatureType.SIMILARITY: 1,  # Single similarity score
    PoseFeatureType.SYMMETRY: 1,    # Single symmetry value
}


# Export all public symbols
__all__: list[str] = [
    # Feature classes
    "PosePointData",
    "PoseAngleData",
    "PoseAngleSimilarityData",
    "PoseAngleSymmetryData",
    "PoseSimilarityBatch",

    # Base classes
    "PoseAngleFeatureBase",

    # Enums
    "PoseJoint",
    "AngleJoint",
    "PoseFeatureType",
    "FeatureStatistic",

    # Constants
    "POSE_JOINT_NAMES",
    "POSE_NUM_JOINTS",
    "POSE_JOINT_COLORS",
    "ANGLE_JOINT_NAMES",
    "ANGLE_NUM_JOINTS",

    # Type aliases
    "PoseFeatureData",
    "PoseSimilarityBatchCallback",
]