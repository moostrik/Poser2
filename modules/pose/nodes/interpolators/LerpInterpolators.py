"""Pose linear interpolation filters for predictable motion interpolation.

Provides true linear interpolation for angles, points, and deltas over exactly
one input interval. Creates predictable, constant-rate movement that completes
when the next target arrives.

Thread Safety:
--------------
Designed for multi-threaded operation:
- Input thread: Calls submit() at ~30 FPS
- Render thread: Calls update() at 60+ FPS

VectorLerp classes are NOT thread-safe. FeatureLerpInterpolator
serializes all access using a lock to prevent concurrent set_target() and update() calls.
"""
from collections import defaultdict

# Pose imports
from modules.pose.features import Angles, BBox, Points2D, AngleSymmetry
from modules.pose.nodes._utils.VectorLerp import AngleLerp, PointLerp, VectorLerp
from modules.pose.nodes.interpolators.BaseInterpolator import FeatureInterpolatorBase
from modules.pose.nodes.Nodes import NodeConfigBase
from modules.pose.Pose import Pose, PoseField
from typing import cast


class LerpInterpolatorConfig(NodeConfigBase):
    """Configuration for pose linear interpolation with automatic change notification."""

    def __init__(self, input_frequency: float = 30.0) -> None:
        super().__init__()
        self.input_frequency: float = input_frequency


class FeatureLerpInterpolator(FeatureInterpolatorBase[LerpInterpolatorConfig]):
    """Generic pose feature linear interpolator."""

    _INTERP_MAP: defaultdict[PoseField, type] = defaultdict(
        lambda: VectorLerp,
        {
            PoseField.angles: AngleLerp,
            PoseField.points: PointLerp,
        }
    )

    def __init__(self, config: LerpInterpolatorConfig, pose_field: PoseField):
        super().__init__(config, pose_field)

    def _create_interpolator(self):
        """Create the underlying lerp interpolator instance."""
        interp_class: type = self._INTERP_MAP[self._pose_field]
        vector_size = len(self._pose_field.get_type().feature_enum())
        clamp_range = self._pose_field.get_type().default_range()
        return interp_class(
            vector_size=vector_size,
            input_frequency=self._config.input_frequency,
            clamp_range=clamp_range
        )

    def _on_config_changed(self) -> None:
        """Handle configuration changes by updating interpolator parameters."""
        with self._lock:
            # Cast to VectorLerp to access input_frequency property
            cast(VectorLerp, self._interpolator).input_frequency = self._config.input_frequency


# Convenience classes
class BBoxLerpInterpolator(FeatureLerpInterpolator):
    def __init__(self, config: LerpInterpolatorConfig) -> None:
        super().__init__(config, PoseField.bbox)


class PointLerpInterpolator(FeatureLerpInterpolator):
    def __init__(self, config: LerpInterpolatorConfig) -> None:
        super().__init__(config, PoseField.points)


class AngleLerpInterpolator(FeatureLerpInterpolator):
    def __init__(self, config: LerpInterpolatorConfig) -> None:
        super().__init__(config, PoseField.angles)


class AngleVelLerpInterpolator(FeatureLerpInterpolator):
    def __init__(self, config: LerpInterpolatorConfig) -> None:
        super().__init__(config, PoseField.angle_vel)


class AngleSymLerpInterpolator(FeatureLerpInterpolator):
    def __init__(self, config: LerpInterpolatorConfig) -> None:
        super().__init__(config, PoseField.angle_sym)