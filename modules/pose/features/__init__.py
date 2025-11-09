"""Pose feature extraction and analysis.

This module provides classes for computing and analyzing pose features:
- PosePoints: Raw keypoint coordinates and scores
- PoseAngles: Joint angles computed from keypoints
- PoseAngleSimilarity: Similarity scores between pose pairs
- PoseAngleSymmetry: Symmetry analysis of poses
"""

from enum import Enum
from typing import Union


from modules.pose.features.base.BaseFeature import (
    BaseFeature,
    NORMALIZED_RANGE,
)

from modules.pose.features.base.NormalizedScalarFeature import (
    AggregationMethod,
)


# Import all feature classes
from modules.pose.features.Point2DFeature import (
    Point2DFeature,
    PointLandmark,
    POINT_LANDMARK_NAMES,
    POINT_NUM_LANDMARKS,
    POINT2D_COORD_RANGE,
)

from modules.pose.features.AngleFeature import (
    AngleFeature,
    AngleLandmark,
    ANGLE_LANDMARK_NAMES,
    ANGLE_NUM_LANDMARKS,
    ANGLE_RANGE,
)

from modules.pose.features.factories.AngleFactory import (
    AngleFactory,
    ANGLE_KEYPOINTS,
)

from modules.pose.features.SimilarityFeature import (
    SimilarityFeature,
    SimilarityBatch,
    SimilarityBatchCallback,
)

from modules.pose.features.SymmetryFeature import (
    SymmetryFeature,
    SymmetricJoint,
    SYMM_JOINT_NAMES,
    SYMM_NUM_JOINTS,
)

from modules.pose.features.factories.SymmetryFactory import (
    SymmetryFactory,
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
    Point2DFeature,
    AngleFeature,
    SymmetryFeature,
]

POSE_FEATURE_CLASSES: dict[PoseFeatureType, type] = {
    PoseFeatureType.POINTS: Point2DFeature,
    PoseFeatureType.ANGLES: AngleFeature,
    PoseFeatureType.DELTA: AngleFeature,
    PoseFeatureType.SYMMETRY: SymmetryFeature,
}

POSE_FEATURE_RANGES: dict[PoseFeatureType, tuple[float, float]] = {
    PoseFeatureType.POINTS: POINT2D_COORD_RANGE,
    PoseFeatureType.ANGLES: ANGLE_RANGE,
    PoseFeatureType.DELTA: ANGLE_RANGE,
    PoseFeatureType.SYMMETRY: NORMALIZED_RANGE,
}

POSE_FEATURE_DIMENSIONS: dict[PoseFeatureType, int] = {
    PoseFeatureType.POINTS: 2,      # (x, y) coordinates
    PoseFeatureType.ANGLES: 1,      # Single angle value
    PoseFeatureType.DELTA: 1,       # Single delta angle value
    PoseFeatureType.SYMMETRY: 1,    # Single symmetry value
}

POSE_CLASS_TO_FEATURE_TYPE: dict[type, PoseFeatureType] = {
    Point2DFeature: PoseFeatureType.POINTS,
    AngleFeature: PoseFeatureType.ANGLES,
    SymmetryFeature: PoseFeatureType.SYMMETRY,
}