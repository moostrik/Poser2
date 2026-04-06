"""Pose smoothing filters using rate limiting for controlled motion."""

from collections import defaultdict

import numpy as np

from modules.pose.features import Angles, AngleVelocity, AngleMotion, AngleSymmetry, BBox, Points2D, Similarity
from modules.pose.features.base import BaseFeature
from modules.pose.nodes._utils.ArrayRateLimit import RateLimit, AngleRateLimit, PointRateLimit, ArrayRateLimit
from modules.pose.nodes.Nodes import FilterNode
from modules.pose.frame import Frame, replace
from modules.settings import Settings, Field


class RateLimiterSettings(Settings):
    """Configuration for pose rate limiting."""
    max_increase: Field[float] = Field(1.0)
    max_decrease: Field[float] = Field(1.0)


class FeatureRateLimiter(FilterNode):
    """Generic pose feature smoother using asymmetric rate limiting."""

    _FEATURE_LIMIT_MAP: defaultdict[type[BaseFeature], type[ArrayRateLimit]] = defaultdict(
        lambda: RateLimit,
        {
            Angles: AngleRateLimit,
            Points2D: PointRateLimit,
        }
    )

    def __init__(self, config: RateLimiterSettings, feature_type: type[BaseFeature]) -> None:
        self._config: RateLimiterSettings = config
        self._feature_type: type[BaseFeature] = feature_type
        limiter_cls = self._FEATURE_LIMIT_MAP[feature_type]
        self._limiter: ArrayRateLimit = limiter_cls(
            vector_size=feature_type.length(),
            max_increase=config.max_increase,
            max_decrease=config.max_decrease,
            clamp_range= feature_type.range()
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
        feature_data = pose[self._feature_type]
        self._limiter.update(feature_data.values)
        limited_values: np.ndarray = self._limiter.value
        limited_data = type(feature_data)(values=limited_values, scores=feature_data.scores)
        return replace(pose, {self._feature_type: limited_data})

    def reset(self) -> None:
        self._limiter.reset()


# Convenience classes
class BBoxRateLimiter(FeatureRateLimiter):
    def __init__(self, config: RateLimiterSettings) -> None:
        super().__init__(config, BBox)


class PointRateLimiter(FeatureRateLimiter):
    def __init__(self, config: RateLimiterSettings) -> None:
        super().__init__(config, Points2D)


class AngleRateLimiter(FeatureRateLimiter):
    def __init__(self, config: RateLimiterSettings) -> None:
        super().__init__(config, Angles)


class AngleVelRateLimiter(FeatureRateLimiter):
    def __init__(self, config: RateLimiterSettings) -> None:
        super().__init__(config, AngleVelocity)


class AngleSymRateLimiter(FeatureRateLimiter):
    def __init__(self, config: RateLimiterSettings) -> None:
        super().__init__(config, AngleSymmetry)


class SimilarityRateLimiter(FeatureRateLimiter):
    def __init__(self, config: RateLimiterSettings) -> None:
        super().__init__(config, Similarity)

class AngleMotionRateLimiter(FeatureRateLimiter):
    def __init__(self, config: RateLimiterSettings) -> None:
        super().__init__(config, AngleMotion)