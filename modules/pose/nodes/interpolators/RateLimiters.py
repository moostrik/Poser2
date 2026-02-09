"""Pose rate limiting filters for constraining maximum change rates.

Provides asymmetric rate limiting for angles, points, and other features
to prevent sudden jumps while allowing different acceleration/deceleration rates.

This is a stateful filter (like StickyFiller) - it processes each input immediately
with rate constraints based on the time since last update.
"""

from dataclasses import replace
from typing import cast
from collections import defaultdict

import numpy as np

# Pose imports
from modules.pose.features import PoseFeatureType, Angles, BBox, Points2D, AngleSymmetry
from modules.pose.nodes._utils.ArrayRateLimit import RateLimit, AngleRateLimit, PointRateLimit
from modules.pose.nodes.Nodes import FilterNode, NodeConfigBase
from modules.pose.Frame import Frame, FrameField


class RateLimiterConfig(NodeConfigBase):
    """Configuration for rate limiting with automatic change notification."""

    def __init__(self, max_increase: float = 1.0, max_decrease: float = 1.0) -> None:
        """
        Args:
            max_increase: Maximum allowed increase per second (units/s)
            max_decrease: Maximum allowed decrease per second (units/s)
        """
        super().__init__()
        self.max_increase: float = max_increase
        self.max_decrease: float = max_decrease


class FeatureRateLimiter(FilterNode):
    """Generic pose feature rate limiter (stateful filter)."""

    _LIMITER_MAP: defaultdict[FrameField, type] = defaultdict(
        lambda: RateLimit,
        {
            FrameField.angles: AngleRateLimit,
            FrameField.points: PointRateLimit,
        }
    )

    def __init__(self, config: RateLimiterConfig, pose_field: FrameField) -> None:
        self._config: RateLimiterConfig = config
        self._pose_field: FrameField = pose_field
        self._limiter: RateLimit = self._create_limiter()
        self._config.add_listener(self._on_config_changed)

    def __del__(self):
        """Cleanup config listener."""
        try:
            self._config.remove_listener(self._on_config_changed)
        except (AttributeError, ValueError):
            pass

    def _create_limiter(self) -> RateLimit:
        """Create the underlying rate limiter instance."""
        limiter_cls = self._LIMITER_MAP[self._pose_field]
        vector_size = len(self._pose_field.get_type().enum())
        clamp_range = self._pose_field.get_type().range()
        return limiter_cls(
            vector_size=vector_size,
            max_increase=self._config.max_increase,
            max_decrease=self._config.max_decrease,
            clamp_range=clamp_range
        )

    def _on_config_changed(self) -> None:
        """Handle configuration changes by updating limiter parameters."""
        self._limiter.max_increase = self._config.max_increase
        self._limiter.max_decrease = self._config.max_decrease

    @property
    def config(self) -> RateLimiterConfig:
        return self._config

    def process(self, pose: Frame) -> Frame:
        """Apply rate limiting to pose feature and return limited pose."""
        feature_data: PoseFeatureType = pose.get_feature(self._pose_field)

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
        return replace(pose, **{self._pose_field.name: limited_data})

    def reset(self) -> None:
        """Reset the limiter's internal state."""
        self._limiter.reset()


# Convenience classes
class BBoxRateLimiter(FeatureRateLimiter):
    def __init__(self, config: RateLimiterConfig) -> None:
        super().__init__(config, FrameField.bbox)


class PointRateLimiter(FeatureRateLimiter):
    def __init__(self, config: RateLimiterConfig) -> None:
        super().__init__(config, FrameField.points)


class AngleRateLimiter(FeatureRateLimiter):
    def __init__(self, config: RateLimiterConfig) -> None:
        super().__init__(config, FrameField.angles)


class AngleVelRateLimiter(FeatureRateLimiter):
    def __init__(self, config: RateLimiterConfig) -> None:
        super().__init__(config, FrameField.angle_vel)


class AngleSymRateLimiter(FeatureRateLimiter):
    def __init__(self, config: RateLimiterConfig) -> None:
        super().__init__(config, FrameField.angle_sym)