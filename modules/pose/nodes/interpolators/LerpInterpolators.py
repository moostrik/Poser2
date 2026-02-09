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
from modules.pose.nodes._utils.ArrayLerp import AngleLerp, PointLerp, Lerp
from modules.pose.nodes.interpolators.BaseInterpolator import FeatureInterpolatorBase, InterpolatorConfigBase
from modules.pose.Frame import Frame, FrameField
from typing import cast


class LerpInterpolatorConfig(InterpolatorConfigBase):
    """Configuration for pose linear interpolation."""
    pass


class FeatureLerpInterpolator(FeatureInterpolatorBase[LerpInterpolatorConfig]):
    """Generic pose feature linear interpolator."""

    _INTERP_MAP: defaultdict[FrameField, type] = defaultdict(
        lambda: Lerp,
        {
            FrameField.angles: AngleLerp,
            FrameField.points: PointLerp,
        }
    )

    def __init__(self, config: LerpInterpolatorConfig, pose_field: FrameField):
        super().__init__(config, pose_field)

    def _create_interpolator(self):
        """Create the underlying lerp interpolator instance."""
        interp_class: type = self._INTERP_MAP[self._pose_field]
        vector_size = len(self._pose_field.get_type().enum())
        clamp_range = self._pose_field.get_type().range()
        return interp_class(
            vector_size=vector_size,
            input_frequency=self._config.input_frequency,
            clamp_range=clamp_range
        )

    def _on_config_changed(self) -> None:
        """Handle configuration changes by updating interpolator parameters."""
        with self._lock:
            # Cast to VectorLerp to access input_frequency property
            cast(Lerp, self._interpolator).input_frequency = self._config.input_frequency
            self._output_interval = 1.0 / self._config.output_frequency


# Convenience classes
class BBoxLerpInterpolator(FeatureLerpInterpolator):
    def __init__(self, config: LerpInterpolatorConfig) -> None:
        super().__init__(config, FrameField.bbox)


class PointLerpInterpolator(FeatureLerpInterpolator):
    def __init__(self, config: LerpInterpolatorConfig) -> None:
        super().__init__(config, FrameField.points)


class AngleLerpInterpolator(FeatureLerpInterpolator):
    def __init__(self, config: LerpInterpolatorConfig) -> None:
        super().__init__(config, FrameField.angles)


class AngleVelLerpInterpolator(FeatureLerpInterpolator):
    def __init__(self, config: LerpInterpolatorConfig) -> None:
        super().__init__(config, FrameField.angle_vel)


class AngleSymLerpInterpolator(FeatureLerpInterpolator):
    def __init__(self, config: LerpInterpolatorConfig) -> None:
        super().__init__(config, FrameField.angle_sym)