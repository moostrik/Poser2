"""Pose smoothing filters using rate limiting for controlled motion."""

from dataclasses import replace
# from typing import Type

import numpy as np

from modules.pose.features.base import BaseFeature
from modules.pose.nodes._utils.VectorRateLimit import VectorRateLimit, AngleRateLimit, PointRateLimit, RateLimit
from modules.pose.nodes._utils.FeatureTypeDispatch import dispatch_by_feature_type
from modules.pose.nodes.Nodes import FilterNode, NodeConfigBase
from modules.pose.Pose import Pose, PoseField


class RateLimitSmootherConfig(NodeConfigBase):
    """Configuration for pose rate limiting with validated parameter ranges."""

    _PARAM_RANGES = {
        'max_increase': (0.0, np.pi),
        'max_decrease': (0.0, np.pi),
    }

    def __init__(self, max_increase: float = 1.0, max_decrease: float = 1.0) -> None:
        super().__init__()
        self.max_increase = max_increase  # Automatically clamped by __setattr__
        self.max_decrease = max_decrease


class FeatureRateLimitSmoother(FilterNode):
    """Generic pose feature smoother using asymmetric rate limiting."""

    def __init__(self, config: RateLimitSmootherConfig, pose_field: PoseField) -> None:
        if not pose_field.is_feature():
            raise ValueError(f"PoseField '{pose_field.value}' is not a feature field")

        self._config: RateLimitSmootherConfig = config
        self._pose_field: PoseField = pose_field

        feature_class: type[BaseFeature] = pose_field.get_feature_class()

        # Dispatch to appropriate limiter class
        limiter_cls: type[RateLimit] = dispatch_by_feature_type(
            feature_class,
            point_handler=PointRateLimit,
            angle_handler=AngleRateLimit,
            scalar_handler=VectorRateLimit
        )

        self._limiter: RateLimit = limiter_cls(
            vector_size=len(feature_class.feature_enum().__members__),
            max_increase=config.max_increase,
            max_decrease=config.max_decrease,
            clamp_range=feature_class.default_range()
        )
        self._config.add_listener(self._on_config_changed)

    def __del__(self):
        try:
            self._config.remove_listener(self._on_config_changed)
        except (AttributeError, ValueError):
            pass

    def _on_config_changed(self) -> None:
        self._limiter.max_increase = self._config.max_increase
        self._limiter.max_decrease = self._config.max_decrease

    @property
    def config(self) -> RateLimitSmootherConfig:
        return self._config

    def process(self, pose: Pose) -> Pose:
        feature_data = pose.get_feature(self._pose_field)
        self._limiter.set_target(feature_data.values)
        self._limiter.update()
        limited_values: np.ndarray = self._limiter.value
        limited_data = type(feature_data)(values=limited_values, scores=feature_data.scores)
        return replace(pose, **{self._pose_field.value: limited_data})

    def reset(self) -> None:
        self._limiter.reset()


# Convenience classes
class AngleRateLimitSmoother(FeatureRateLimitSmoother):
    def __init__(self, config: RateLimitSmootherConfig) -> None:
        super().__init__(config, PoseField.angles)


class DeltaRateLimitSmoother(FeatureRateLimitSmoother):
    def __init__(self, config: RateLimitSmootherConfig) -> None:
        super().__init__(config, PoseField.deltas)


class BBoxRateLimitSmoother(FeatureRateLimitSmoother):
    def __init__(self, config: RateLimitSmootherConfig) -> None:
        super().__init__(config, PoseField.bbox)


class PointRateLimitSmoother(FeatureRateLimitSmoother):
    def __init__(self, config: RateLimitSmootherConfig) -> None:
        super().__init__(config, PoseField.points)


class SymmetryRateLimitSmoother(FeatureRateLimitSmoother):
    def __init__(self, config: RateLimitSmootherConfig) -> None:
        super().__init__(config, PoseField.symmetry)