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

from modules.pose.features.base.BaseScalarFeature import (
    BaseScalarFeature,
)

from modules.pose.features.base.NormalizedScalarFeature import (
    AggregationMethod,
    NormalizedScalarFeature,
)

from modules.pose.features.BBoxFeature import (
    BBoxFeature,
    BBoxElement,
    BBOX_ELEMENT_NAMES,
    BBOX_NUM_ELEMENTS,
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

# Import all feature classes
from modules.pose.features.Point2DFeature import (
    Point2DFeature,
    PointLandmark,
    POINT_LANDMARK_NAMES,
    POINT_NUM_LANDMARKS,
    POINT2D_COORD_RANGE,
)

from modules.pose.features.SimilarityFeature import (
    SimilarityFeature,
    SimilarityBatch,
    SimilarityBatchCallback,
)

from modules.pose.features.SymmetryFeature import (
    SymmetryFeature,
    SymmetryElement,
    SYMMETRY_ELEMENT_NAMES,
    SYMMETRY_NUM_ELEMENTS,
)

from modules.pose.features.factories.SymmetryFactory import (
    SymmetryFactory,
)

# Feature type enum for dynamic dispatch
class PoseFeatureType(Enum):
    """Enum for different pose feature types."""
    ANGLES = "angles"
    BBOX = "bbox"
    DELTA = "delta"
    POINTS = "points"
    SYMMETRY = "symmetry"

# Type alias for any feature data
PoseFeature = Union[
    AngleFeature,
    BBoxFeature,
    Point2DFeature,
    SymmetryFeature,
]

POSE_FEATURE_CLASSES: dict[PoseFeatureType, type] = {
    PoseFeatureType.ANGLES: AngleFeature,
    PoseFeatureType.BBOX: BBoxFeature,
    PoseFeatureType.DELTA: AngleFeature,
    PoseFeatureType.POINTS: Point2DFeature,
    PoseFeatureType.SYMMETRY: SymmetryFeature,
}

POSE_FEATURE_RANGES: dict[PoseFeatureType, tuple[float, float]] = {
    PoseFeatureType.ANGLES: AngleFeature.default_range(),
    PoseFeatureType.BBOX: BBoxFeature.default_range(),
    PoseFeatureType.DELTA: AngleFeature.default_range(),
    PoseFeatureType.POINTS: Point2DFeature.default_range(),
    PoseFeatureType.SYMMETRY: SymmetryFeature.default_range(),
}

POSE_FEATURE_DIMENSIONS: dict[PoseFeatureType, int] = {
    PoseFeatureType.POINTS: 2,      # (x, y) coordinates
    PoseFeatureType.ANGLES: 1,      # Single angle value
    PoseFeatureType.DELTA: 1,       # Single delta angle value
    PoseFeatureType.SYMMETRY: 1,    # Single symmetry value
}

POSE_CLASS_TO_FEATURE_TYPE: dict[type, PoseFeatureType] = {
    AngleFeature: PoseFeatureType.ANGLES,
    BBoxFeature: PoseFeatureType.BBOX,
    AngleFeature: PoseFeatureType.DELTA,
    Point2DFeature: PoseFeatureType.POINTS,
    SymmetryFeature: PoseFeatureType.SYMMETRY,
}