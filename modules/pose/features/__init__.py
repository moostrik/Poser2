from enum import IntEnum

from .base import           BaseFeature, BaseScalarFeature, BaseVectorFeature, NormalizedScalarFeature, AggregationMethod, NormalizedSingleValue, SingleValue
from .BBox import           BBox, BBoxElement
from .Points2D import       Points2D, PointLandmark
from .Angles import         Angles, AngleLandmark
from .AngleVelocity import  AngleVelocity
from .AngleMotion import    AngleMotion
from .AngleSymmetry import  AngleSymmetry, SymmetryElement
from .Similarity import     Similarity
from .LeaderScore import    LeaderScore
from .MotionGate import     MotionGate
from .MotionTime import     MotionTime
from .Age import            Age


FEATURES: list[type[BaseFeature]] = [
    Age, AngleMotion, Angles, AngleSymmetry, AngleVelocity,
    BBox, LeaderScore, MotionGate, MotionTime, Points2D, Similarity,
]

Feature: type[IntEnum] = IntEnum('Feature', {cls.__name__: i for i, cls in enumerate(FEATURES, 1)})  # type: ignore[misc]
"""Enum with one member per concrete feature class, auto-generated from FEATURES."""

FEATURE_CLASS: dict = dict(zip(Feature, FEATURES))  # type: ignore[arg-type]
"""Map Feature enum member → concrete feature class."""

SCALAR_FEATURES: list[type[BaseScalarFeature]] = [
    f for f in FEATURES if issubclass(f, BaseScalarFeature)
]

TRACK_FEATURES: set[type[BaseScalarFeature]] = {Similarity, LeaderScore, MotionGate}
"""Scalar features whose elements are indexed by track (pose), not by joint/body part."""


def configure_features(max_poses: int) -> None:
    """Configure all track-indexed features at startup.

    Must be called once before building the pipeline so that every feature
    can produce data (or NaN dummies). Idempotent — only the first call
    has effect.
    """
    for ft in TRACK_FEATURES:
        ft.configure(max_poses)  # type: ignore[attr-defined]