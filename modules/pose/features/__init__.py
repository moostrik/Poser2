from typing import Union

from modules.pose.features.base import          BaseFeature, BaseScalarFeature, BaseVectorFeature, NormalizedScalarFeature, AggregationMethod
from modules.pose.features.BBox import          BBox, BBoxElement
from modules.pose.features.Points2D import      Points2D, PointLandmark
from modules.pose.features.Angles import        Angles, AngleLandmark
from modules.pose.features.AngleVelocity import AngleVelocity
from modules.pose.features.AngleSymmetry import AngleSymmetry, SymmetryElement

PoseFeatureType = Union[BBox, Points2D, Angles, AngleVelocity, AngleSymmetry]
