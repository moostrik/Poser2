"""Utility for dispatching feature types to appropriate algorithm classes."""

from typing import Type, TypeVar

from modules.pose.features import Points2D, Angles
from modules.pose.features.base import BaseFeature

T = TypeVar('T')


def dispatch_by_feature_type(
    feature_class: Type[BaseFeature],
    point_handler: Type[T],
    angle_handler: Type[T],
    scalar_handler: Type[T]
) -> Type[T]:
    """Dispatch a feature class to the appropriate handler class.

    Args:
        feature_class: The feature class to dispatch (e.g., Points2D, Angles, BBox)
        point_handler: Handler class for 2D point features
        angle_handler: Handler class for angular features
        scalar_handler: Handler class for scalar features

    Returns:
        The appropriate handler class based on feature type

    Example:
        limiter_cls = dispatch_by_feature_type(
            feature_class,
            PointRateLimit,
            AngleRateLimit,
            VectorRateLimit
        )
    """
    if issubclass(feature_class, Points2D):
        return point_handler
    elif issubclass(feature_class, Angles):
        return angle_handler
    else:
        return scalar_handler