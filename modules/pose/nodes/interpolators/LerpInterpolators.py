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
from modules.pose.features import Angles, BBox, Points2D, AngleVelocity, AngleSymmetry
from modules.pose.features.base import BaseFeature
from modules.pose.nodes._utils.ArrayLerp import AngleLerp, PointLerp, Lerp
from modules.pose.nodes.interpolators.BaseInterpolator import FeatureInterpolatorBase, InterpolatorSettingsBase
from modules.pose.frame import Frame
from typing import cast


class LerpInterpolatorSettings(InterpolatorSettingsBase):
    """Configuration for pose linear interpolation."""
    pass


class FeatureLerpInterpolator(FeatureInterpolatorBase[LerpInterpolatorSettings]):
    """Generic pose feature linear interpolator."""

    _INTERP_MAP: defaultdict[type[BaseFeature], type] = defaultdict(
        lambda: Lerp,
        {
            Angles: AngleLerp,
            Points2D: PointLerp,
        }
    )

    def __init__(self, config: LerpInterpolatorSettings, feature_type: type[BaseFeature]):
        super().__init__(config, feature_type)

    def _create_interpolator(self):
        """Create the underlying lerp interpolator instance."""
        interp_class: type = self._INTERP_MAP[self._feature_type]
        vector_size = self._feature_type.length()
        clamp_range = self._feature_type.range()
        return interp_class(
            vector_size=vector_size,
            input_frequency=self._config.input_frequency,
            clamp_range=clamp_range
        )

    def _on_config_changed(self, _=None) -> None:
        """Handle configuration changes by updating interpolator parameters."""
        with self._lock:
            # Cast to VectorLerp to access input_frequency property
            cast(Lerp, self._interpolator).input_frequency = self._config.input_frequency
            self._output_interval = 1.0 / self._config.output_frequency


# Convenience classes
class BBoxLerpInterpolator(FeatureLerpInterpolator):
    def __init__(self, config: LerpInterpolatorSettings) -> None:
        super().__init__(config, BBox)


class PointLerpInterpolator(FeatureLerpInterpolator):
    def __init__(self, config: LerpInterpolatorSettings) -> None:
        super().__init__(config, Points2D)


class AngleLerpInterpolator(FeatureLerpInterpolator):
    def __init__(self, config: LerpInterpolatorSettings) -> None:
        super().__init__(config, Angles)


class AngleVelLerpInterpolator(FeatureLerpInterpolator):
    def __init__(self, config: LerpInterpolatorSettings) -> None:
        super().__init__(config, AngleVelocity)


class AngleSymLerpInterpolator(FeatureLerpInterpolator):
    def __init__(self, config: LerpInterpolatorSettings) -> None:
        super().__init__(config, AngleSymmetry)