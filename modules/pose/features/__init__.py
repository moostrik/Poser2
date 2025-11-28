from typing import Union

from .base import           BaseFeature, BaseScalarFeature, BaseVectorFeature, NormalizedScalarFeature, AggregationMethod
from .BBox import           BBox, BBoxElement
from .Points2D import       Points2D, PointLandmark
from .Angles import         Angles, AngleLandmark
from .AngleVelocity import  AngleVelocity
from .AngleSymmetry import  AngleSymmetry, SymmetryElement
from .Similarity import     Similarity

PoseFeatureType = Union[BBox, Points2D, Angles, AngleVelocity, AngleSymmetry, Similarity]
