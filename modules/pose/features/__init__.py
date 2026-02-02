from typing import Union

from .base import           BaseFeature, BaseScalarFeature, BaseVectorFeature, NormalizedScalarFeature, AggregationMethod
from .BBox import           BBox, BBoxElement
from .Points2D import       Points2D, PointLandmark
from .Angles import         Angles, AngleLandmark
from .AngleVelocity import  AngleVelocity
from .AngleMotion import    AngleMotion
from .AngleSymmetry import  AngleSymmetry, SymmetryElement
from .Similarity import     Similarity, configure_similarity
from .LeaderScore import    LeaderScore, configure_leader_score
from .SingleValue import    SingleValue

PoseFeatureType = Union[BBox, Points2D, Angles, AngleVelocity, AngleMotion, AngleSymmetry, Similarity, LeaderScore, SingleValue]
