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


def _collect_concrete(cls: type) -> list[type[BaseFeature]]:
    """Recursively collect all concrete (non-abstract, non-private, non-base)
    subclasses of *cls*."""
    result: list[type[BaseFeature]] = []
    for sub in cls.__subclasses__():
        result.extend(_collect_concrete(sub))
        if getattr(sub, '__abstractmethods__', frozenset()):
            continue
        if sub.__name__.startswith('_'):
            continue
        if 'features.base' in sub.__module__:
            continue
        result.append(sub)
    result.sort(key=lambda c: c.__name__)
    return result


FEATURES: list[type[BaseFeature]] = _collect_concrete(BaseFeature)

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