"""Pose smoothing filters using rate limiting for controlled motion."""

from dataclasses import replace
from collections import defaultdict
# from typing import Type

import numpy as np

from modules.pose.features import PoseFeatureType
from modules.pose.nodes._utils.VectorRateLimit import VectorRateLimit, AngleRateLimit, PointRateLimit, RateLimit
from modules.pose.nodes._utils.FeatureTypeDispatch import dispatch_by_feature_type
from modules.pose.nodes.Nodes import FilterNode, NodeConfigBase
from modules.pose.Pose import Pose, PoseField


class RateLimiterConfig(NodeConfigBase):
    """Configuration for pose rate limiting with validated parameter ranges."""

    _PARAM_RANGES = {
        'max_increase': (0.0, np.pi),
        'max_decrease': (0.0, np.pi),
    }

    def __init__(self, max_increase: float = 1.0, max_decrease: float = 1.0) -> None:
        super().__init__()
        self.max_increase = max_increase  # Automatically clamped by __setattr__
        self.max_decrease = max_decrease


class FeatureRateLimiter(FilterNode):
    """Generic pose feature smoother using asymmetric rate limiting."""

    _FEATURE_LIMIT_MAP: defaultdict[PoseField, type[RateLimit]] = defaultdict(
        lambda: VectorRateLimit,
        {
            PoseField.angles: AngleRateLimit,
            PoseField.points: PointRateLimit,
        }
    )

    def __init__(self, config: RateLimiterConfig, pose_field: PoseField) -> None:
        self._config: RateLimiterConfig = config
        self._pose_field: PoseField = pose_field
        limiter_cls = self._FEATURE_LIMIT_MAP[pose_field]
        self._limiter: RateLimit = limiter_cls(
            vector_size=len(pose_field.get_type().feature_enum()),
            max_increase=config.max_increase,
            max_decrease=config.max_decrease,
            clamp_range= pose_field.get_type().default_range()
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
    def config(self) -> RateLimiterConfig:
        return self._config

    def process(self, pose: Pose) -> Pose:
        feature_data = pose.get_feature(self._pose_field)
        self._limiter.set_target(feature_data.values)
        self._limiter.update()
        limited_values: np.ndarray = self._limiter.value
        limited_data = type(feature_data)(values=limited_values, scores=feature_data.scores)
        return replace(pose, **{self._pose_field.name: limited_data})

    def reset(self) -> None:
        self._limiter.reset()


# Convenience classes
class BBoxRateLimiter(FeatureRateLimiter):
    def __init__(self, config: RateLimiterConfig) -> None:
        super().__init__(config, PoseField.bbox)


class PointRateLimiter(FeatureRateLimiter):
    def __init__(self, config: RateLimiterConfig) -> None:
        super().__init__(config, PoseField.points)


class AngleRateLimiter(FeatureRateLimiter):
    def __init__(self, config: RateLimiterConfig) -> None:
        super().__init__(config, PoseField.angles)


class AngleVelRateLimiter(FeatureRateLimiter):
    def __init__(self, config: RateLimiterConfig) -> None:
        super().__init__(config, PoseField.angle_vel)


class AngleSymRateLimiter(FeatureRateLimiter):
    def __init__(self, config: RateLimiterConfig) -> None:
        super().__init__(config, PoseField.angle_sym)