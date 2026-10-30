from enum import IntEnum

from .base import           BaseFeature, BaseScalarFeature, BaseVectorFeature, NormalizedScalarFeature, AggregationMethod, NormalizedSingleValue, SingleValue, SingleAngle, SYMMETRIC_PI_RANGE
from .bbox import           BBox, BBoxElement
from .points_2d import      Points2D, PointLandmark
from .angles import         Angles, AngleLandmark
from .angle_velocity import AngleVelocity
from .angle_motion import   AngleMotion
from .angle_symmetry import AngleSymmetry, SymmetryElement
from .arm_deviation import  ArmDeviation
from .arm_travel import     ArmTravel, TravelElement
from .shoulder_difference import ShoulderDifference
from .similarity import     Similarity
from .leader_score import   LeaderScore
from .motion_gate import    MotionGate
from .motion_time import    MotionTime
from .age import            Age
from .azimuth import        Azimuth
from .bbox_azimuth import   BBoxAzimuth
from .distance import       Distance
from .leg_deviation import  LegDeviation
from .torso_tilt import     TorsoTilt
from .group_similarity import GroupSimilarity
from .group_sync import     GroupSync


FEATURES: list[type[BaseFeature]] = [
    Age, AngleMotion, Angles, AngleSymmetry, AngleVelocity, ArmDeviation, ArmTravel,
    Azimuth, BBox, BBoxAzimuth, Distance, LeaderScore, LegDeviation, MotionGate, MotionTime, Points2D,
    ShoulderDifference, Similarity, TorsoTilt,
]

Feature: type[IntEnum] = IntEnum('Feature', {cls.__name__: i for i, cls in enumerate(FEATURES, 1)})  # type: ignore[misc]
"""Enum with one member per concrete feature class, auto-generated from FEATURES."""

FEATURE_CLASS: dict = dict(zip(Feature, FEATURES))
"""Map Feature enum member → concrete feature class."""

SCALAR_FEATURES: list[type[BaseScalarFeature]] = [
    f for f in FEATURES if issubclass(f, BaseScalarFeature)
]

TRACK_FEATURES: set[type[BaseScalarFeature]] = {Similarity, LeaderScore, MotionGate}
"""Scalar features whose elements are indexed by track (pose), not by joint/body part."""


def configure_features(max_players: int) -> None:
    """Configure all track-indexed features at startup.

    Must be called once before building the pipeline so that every feature
    can produce data (or NaN dummies). Idempotent — only the first call
    has effect.
    """
    for ft in TRACK_FEATURES:
        ft.configure(max_players)  # type: ignore[attr-defined]