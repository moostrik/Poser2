"""Pose smoothing filters using rate limiting for controlled motion."""

from collections import defaultdict

import numpy as np

from ...features import Angles, AngleVelocity, AngleMotion, AngleSymmetry, BBox, Points2D, Similarity, BaseFeature
from .._utils.array_rate_limit import RateLimit, AngleRateLimit, PointRateLimit, ArrayRateLimit
from ..nodes import FilterNode
from ...frame import Frame, replace
from modules.settings import BaseSettings, Field


class RateLimiterSettings(BaseSettings):
    """Configuration for pose rate limiting: the fastest a value may move, per second."""
    max_increase: Field[float] = Field(1.0, min=0.0, description="Fastest rise (units per second)")
    max_decrease: Field[float] = Field(1.0, min=0.0, description="Fastest fall (units per second)")


class FeatureRateLimiter(FilterNode):
    """Generic pose feature smoother using asymmetric rate limiting."""

    _FEATURE_LIMIT_MAP: defaultdict[type[BaseFeature], type[ArrayRateLimit]] = defaultdict(
        lambda: RateLimit,
        {
            Angles: AngleRateLimit,
            Points2D: PointRateLimit,
        }
    )

    def __init__(self, settings:RateLimiterSettings, feature_type: type[BaseFeature]) -> None:
        self.settings: RateLimiterSettings = settings
        self._feature_type: type[BaseFeature] = feature_type
        limiter_cls = self._FEATURE_LIMIT_MAP[feature_type]
        self._limiter: ArrayRateLimit = limiter_cls(
            vector_size=feature_type.length(),
            max_increase=settings.max_increase,
            max_decrease=settings.max_decrease,
            clamp_range= feature_type.range()
        )
        self.settings.bind_all(self._on_settings_changed)

    def __del__(self):
        try:
            self.settings.unbind_all(self._on_settings_changed)
        except (AttributeError, ValueError):
            pass

    def _on_settings_changed(self, _=None) -> None:
        self._limiter.max_increase = self.settings.max_increase
        self._limiter.max_decrease = self.settings.max_decrease

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
    def __init__(self, settings:RateLimiterSettings) -> None:
        super().__init__(settings,BBox)


class PointRateLimiter(FeatureRateLimiter):
    def __init__(self, settings:RateLimiterSettings) -> None:
        super().__init__(settings,Points2D)


class AngleRateLimiter(FeatureRateLimiter):
    def __init__(self, settings:RateLimiterSettings) -> None:
        super().__init__(settings,Angles)


class AngleVelocityRateLimiter(FeatureRateLimiter):
    def __init__(self, settings:RateLimiterSettings) -> None:
        super().__init__(settings,AngleVelocity)


class AngleSymmetryRateLimiter(FeatureRateLimiter):
    def __init__(self, settings:RateLimiterSettings) -> None:
        super().__init__(settings,AngleSymmetry)


class SimilarityRateLimiter(FeatureRateLimiter):
    def __init__(self, settings:RateLimiterSettings) -> None:
        super().__init__(settings,Similarity)

class AngleMotionRateLimiter(FeatureRateLimiter):
    def __init__(self, settings:RateLimiterSettings) -> None:
        super().__init__(settings,AngleMotion)