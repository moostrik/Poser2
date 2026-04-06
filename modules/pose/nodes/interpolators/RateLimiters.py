"""Pose rate limiting filters for constraining maximum change rates.

Provides asymmetric rate limiting for angles, points, and other features
to prevent sudden jumps while allowing different acceleration/deceleration rates.

This is a stateful filter (like StickyFiller) - it processes each input immediately
with rate constraints based on the time since last update.
"""

from typing import cast
from collections import defaultdict

import numpy as np

# Pose imports
from modules.pose.features import Angles, BBox, Points2D, AngleVelocity, AngleSymmetry
from modules.pose.features.base import BaseFeature
from modules.pose.nodes._utils.ArrayRateLimit import RateLimit, AngleRateLimit, PointRateLimit
from modules.pose.nodes.Nodes import FilterNode
from modules.pose.frame import Frame, replace
from modules.settings import Settings, Field


class RateLimiterSettings(Settings):
    """Configuration for rate limiting."""
    max_increase: Field[float] = Field(1.0)
    max_decrease: Field[float] = Field(1.0)


class FeatureRateLimiter(FilterNode):
    """Generic pose feature rate limiter (stateful filter)."""

    _LIMITER_MAP: defaultdict[type[BaseFeature], type] = defaultdict(
        lambda: RateLimit,
        {
            Angles: AngleRateLimit,
            Points2D: PointRateLimit,
        }
    )

    def __init__(self, config: RateLimiterSettings, feature_type: type[BaseFeature]) -> None:
        self._config: RateLimiterSettings = config
        self._feature_type: type[BaseFeature] = feature_type
        self._limiter: RateLimit = self._create_limiter()
        self._config.bind_all(self._on_config_changed)

    def __del__(self):
        """Cleanup config listener."""
        try:
            self._config.unbind_all(self._on_config_changed)
        except (AttributeError, ValueError):
            pass

    def _create_limiter(self) -> RateLimit:
        """Create the underlying rate limiter instance."""
        limiter_cls = self._LIMITER_MAP[self._feature_type]
        vector_size = self._feature_type.length()
        clamp_range = self._feature_type.range()
        return limiter_cls(
            vector_size=vector_size,
            max_increase=self._config.max_increase,
            max_decrease=self._config.max_decrease,
            clamp_range=clamp_range
        )

    def _on_config_changed(self, _=None) -> None:
        """Handle configuration changes by updating limiter parameters."""
        self._limiter.max_increase = self._config.max_increase
        self._limiter.max_decrease = self._config.max_decrease

    @property
    def config(self) -> RateLimiterSettings:
        return self._config

    def process(self, pose: Frame) -> Frame:
        """Apply rate limiting to pose feature and return limited pose."""
        feature_data = pose[self._feature_type]

        # Update limiter with new values (uses internal time tracking)
        self._limiter.update(feature_data.values)
        limited_values = self._limiter.value

        # Create limited feature data with adjusted scores
        if limited_values.ndim > 1:
            has_nan = np.any(np.isnan(limited_values), axis=-1)
        else:
            has_nan = np.isnan(limited_values)
        limited_scores = np.where(has_nan, 0.0, feature_data.scores).astype(np.float32)

        limited_data = type(feature_data)(values=limited_values, scores=limited_scores)
        return replace(pose, {self._feature_type: limited_data})

    def reset(self) -> None:
        """Reset the limiter's internal state."""
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