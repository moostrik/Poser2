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

PoseFeature = Union[
    AngleFeature,
    BBoxFeature,
    Point2DFeature,
    SymmetryFeature,
]
