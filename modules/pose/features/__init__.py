from typing import Union

from modules.pose.features.BBox import       BBox, BBoxElement
from modules.pose.features.Angles import      Angles, AngleLandmark
from modules.pose.features.Points2D import    Points2D, PointLandmark
from modules.pose.features.Symmetry import   Symmetry, SymmetryElement, AggregationMethod

PoseFeature = Union[Angles, BBox, Points2D, Symmetry]
