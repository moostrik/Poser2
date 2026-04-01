"""Pose smoothing filters using rate limiting for controlled motion."""

from dataclasses import replace
from collections import defaultdict
# from typing import Type

import numpy as np

from modules.pose.features import PoseFeatureType
from modules.pose.nodes._utils.ArrayRateLimit import RateLimit, AngleRateLimit, PointRateLimit, ArrayRateLimit
from modules.pose.nodes.Nodes import FilterNode
from modules.pose.Frame import Frame, FrameField
from modules.settings import Settings, Field


class RateLimiterSettings(Settings):
    """Configuration for pose rate limiting."""
    max_increase: Field[float] = Field(1.0)
    max_decrease: Field[float] = Field(1.0)


class FeatureRateLimiter(FilterNode):
    """Generic pose feature smoother using asymmetric rate limiting."""

    _FEATURE_LIMIT_MAP: defaultdict[FrameField, type[ArrayRateLimit]] = defaultdict(
        lambda: RateLimit,
        {
            FrameField.angles: AngleRateLimit,
            FrameField.points: PointRateLimit,
        }
    )

    def __init__(self, config: RateLimiterSettings, pose_field: FrameField) -> None:
        self._config: RateLimiterSettings = config
        self._pose_field: FrameField = pose_field
        limiter_cls = self._FEATURE_LIMIT_MAP[pose_field]
        self._limiter: ArrayRateLimit = limiter_cls(
            vector_size=len(pose_field.get_type().enum()),
            max_increase=config.max_increase,
            max_decrease=config.max_decrease,
            clamp_range= pose_field.get_type().range()
        )
        self._config.bind_all(self._on_config_changed)

    def __del__(self):
        try:
            self._config.unbind_all(self._on_config_changed)
        except (AttributeError, ValueError):
            pass

    def _on_config_changed(self, _=None) -> None:
        self._limiter.max_increase = self._config.max_increase
        self._limiter.max_decrease = self._config.max_decrease

    @property
    def config(self) -> RateLimiterSettings:
        return self._config

    def process(self, pose: Frame) -> Frame:
        feature_data = pose.get_feature(self._pose_field)
        self._limiter.update(feature_data.values)
        limited_values: np.ndarray = self._limiter.value
        limited_data = type(feature_data)(values=limited_values, scores=feature_data.scores)
        return replace(pose, **{self._pose_field.name: limited_data})

    def reset(self) -> None:
        self._limiter.reset()


# Convenience classes
class BBoxRateLimiter(FeatureRateLimiter):
    def __init__(self, config: RateLimiterSettings) -> None:
        super().__init__(config, FrameField.bbox)


class PointRateLimiter(FeatureRateLimiter):
    def __init__(self, config: RateLimiterSettings) -> None:
        super().__init__(config, FrameField.points)


class AngleRateLimiter(FeatureRateLimiter):
    def __init__(self, config: RateLimiterSettings) -> None:
        super().__init__(config, FrameField.angles)


class AngleVelRateLimiter(FeatureRateLimiter):
    def __init__(self, config: RateLimiterSettings) -> None:
        super().__init__(config, FrameField.angle_vel)


class AngleSymRateLimiter(FeatureRateLimiter):
    def __init__(self, config: RateLimiterSettings) -> None:
        super().__init__(config, FrameField.angle_sym)


class SimilarityRateLimiter(FeatureRateLimiter):
    def __init__(self, config: RateLimiterSettings) -> None:
        super().__init__(config, FrameField.similarity)

class AngleMotionRateLimiter(FeatureRateLimiter):
    def __init__(self, config: RateLimiterSettings) -> None:
        super().__init__(config, FrameField.angle_motion)