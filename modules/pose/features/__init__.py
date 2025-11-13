from typing import Union

from modules.pose.features.BBoxFeature import       BBoxFeature, BBoxElement
from modules.pose.features.AngleFeature import      AngleFeature, AngleLandmark
from modules.pose.features.Point2DFeature import    Point2DFeature, PointLandmark
from modules.pose.features.SymmetryFeature import   SymmetryFeature, SymmetryElement, AggregationMethod

PoseFeature = Union[AngleFeature, BBoxFeature, Point2DFeature, SymmetryFeature]
