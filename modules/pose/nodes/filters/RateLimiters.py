"""Pose rate limiting filters for constraining maximum change rates.

Provides asymmetric rate limiting for angles, points, and other features
to prevent sudden jumps while allowing different acceleration/deceleration rates.
"""

# Standard library imports
from dataclasses import replace

# Third-party imports
import numpy as np

# Pose imports
from modules.pose.features import Angles, BBox, Points2D, Symmetry
from modules.pose.nodes._utils.VectorRateLimit import VectorRateLimit, AngleRateLimit, PointRateLimit
from modules.pose.nodes.Nodes import InterpolatorNode, NodeConfigBase
from modules.pose.Pose import Pose


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


class FeatureRateLimiter(InterpolatorNode):
    """Generic pose feature rate limiter.

    Args:
        config: Rate limiter configuration
        feature_class: Feature class type (e.g., Angles, Points2D)
        attr_name: Name of the pose attribute to limit

    Example:
        limiter = FeatureRateLimiter(config, Angles, "angles")
        limiter = FeatureRateLimiter(config, Points2D, "points")
    """

    # Registry mapping feature classes to rate limiter classes
    LIMITER_REGISTRY = {
        Angles: AngleRateLimit,
        BBox: VectorRateLimit,
        Points2D: PointRateLimit,
        Symmetry: VectorRateLimit,
    }

    def __init__(self, config: RateLimiterConfig, feature_class: type, attr_name: str):
        if feature_class not in self.LIMITER_REGISTRY:
            valid_classes = [cls.__name__ for cls in self.LIMITER_REGISTRY.keys()]
            raise ValueError(
                f"Unknown feature class '{feature_class.__name__}'. "
                f"Must be one of: {valid_classes}"
            )

        self._config = config
        self._attr_name = attr_name
        self._feature_class = feature_class
        self._last_pose: Pose | None = None

        limiter_cls = self.LIMITER_REGISTRY[feature_class]
        self._limiter = limiter_cls(
            vector_size=len(feature_class.feature_enum()),
            max_increase=config.max_increase,
            max_decrease=config.max_decrease,
            clamp_range=feature_class.default_range()
        )
        self._config.add_listener(self._on_config_changed)

    def __del__(self):
        """Cleanup config listener to prevent memory leaks."""
        try:
            self._config.remove_listener(self._on_config_changed)
        except (AttributeError, ValueError):
            pass  # Config already cleaned up or listener not found

    def _on_config_changed(self) -> None:
        self._limiter.max_increase = self._config.max_increase
        self._limiter.max_decrease = self._config.max_decrease

    @property
    def config(self) -> RateLimiterConfig:
        return self._config

    @property
    def attr_name(self) -> str:
        return self._attr_name

    def submit(self, pose: Pose) -> None:
        """Submit target pose (called at input frequency)."""
        feature_data = getattr(pose, self._attr_name)
        self._limiter.add_sample(feature_data.values)
        self._last_pose = pose

    def update(self, time_stamp: float | None = None) -> Pose | None:
        """Get rate-limited pose (called at render frequency)."""
        if self._last_pose is None:
            return None

        self._limiter.update(time_stamp)
        limited_values: np.ndarray = self._limiter.value

        feature_data = getattr(self._last_pose, self._attr_name)
        limited_data = type(feature_data)(values=limited_values, scores=feature_data.scores)
        return replace(self._last_pose, **{self._attr_name: limited_data})

    def reset(self) -> None:
        self._limiter.reset()
        self._last_pose = None


# Convenience classes
class AngleRateLimiter(FeatureRateLimiter):
    def __init__(self, config: RateLimiterConfig) -> None:
        super().__init__(config, Angles, "angles")


class DeltaRateLimiter(FeatureRateLimiter):
    def __init__(self, config: RateLimiterConfig) -> None:
        super().__init__(config, Angles, "deltas")


class BBoxRateLimiter(FeatureRateLimiter):
    def __init__(self, config: RateLimiterConfig) -> None:
        super().__init__(config, BBox, "bbox")


class PointRateLimiter(FeatureRateLimiter):
    def __init__(self, config: RateLimiterConfig) -> None:
        super().__init__(config, Points2D, "points")


class SymmetryRateLimiter(FeatureRateLimiter):
    def __init__(self, config: RateLimiterConfig) -> None:
        super().__init__(config, Symmetry, "symmetry")