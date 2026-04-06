from .base import           BaseFeature, BaseScalarFeature, BaseVectorFeature, NormalizedScalarFeature, AggregationMethod, NormalizedSingleValue, SingleValue
from .BBox import           BBox, BBoxElement
from .Points2D import       Points2D, PointLandmark
from .Angles import         Angles, AngleLandmark
from .AngleVelocity import  AngleVelocity
from .AngleMotion import    AngleMotion
from .AngleSymmetry import  AngleSymmetry, SymmetryElement
from .Similarity import     Similarity, configure_similarity
from .LeaderScore import    LeaderScore, configure_leader_score
from .MotionGate import     MotionGate, configure_motion_gate
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