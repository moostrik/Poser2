"""Pose rate limiting filters for constraining maximum change rates.

Provides asymmetric rate limiting for angles, points, and other features
to prevent sudden jumps while allowing different acceleration/deceleration rates.

Thread Safety:
--------------
Designed for multi-threaded operation:
- Input thread: Calls submit() at ~30 FPS
- Render thread: Calls update() at 60+ FPS

VectorRateLimit classes are NOT thread-safe. FeatureRateLimiter
serializes all access using a lock to prevent concurrent set_target() and update() calls.
"""

from typing import cast
from collections import defaultdict

# Pose imports
from modules.pose.features import Angles, BBox, Points2D, AngleSymmetry
from modules.pose.nodes._utils.VectorRateLimit import VectorRateLimit, AngleRateLimit, PointRateLimit
from modules.pose.nodes.interpolators.BaseInterpolator import FeatureInterpolatorBase
from modules.pose.nodes.Nodes import NodeConfigBase
from modules.pose.nodes.interpolators.BaseInterpolator import FeatureInterpolatorBase
from modules.pose.Pose import Pose, PoseField


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


class FeatureRateLimiter(FeatureInterpolatorBase[RateLimiterConfig]):
    """Generic pose feature rate limiter."""

    _INTERP_MAP: defaultdict[PoseField, type] = defaultdict(
        lambda: VectorRateLimit,
        {
            PoseField.angles: AngleRateLimit,
            PoseField.points: PointRateLimit,
        }
    )

    def _create_interpolator(self):
        """Create the underlying rate limiter instance."""
        interpolator_cls = self._INTERP_MAP[self._pose_field]
        vector_size = len(self._pose_field.get_type().feature_enum())
        clamp_range = self._pose_field.get_type().default_range()
        return interpolator_cls(
            vector_size=vector_size,
            max_increase=self._config.max_increase,
            max_decrease=self._config.max_decrease,
            clamp_range=clamp_range
        )

    def _on_config_changed(self) -> None:
        """Handle configuration changes by updating limiter parameters."""
        with self._lock:
            cast(VectorRateLimit, self._interpolator).max_increase = self._config.max_increase
            cast(VectorRateLimit, self._interpolator).max_decrease = self._config.max_decrease


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