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

# Pose imports
from modules.pose.features import Angles, BBox, Points2D, Symmetry
from modules.pose.nodes._utils.VectorLerp import AngleLerp, PointLerp, VectorLerp
from modules.pose.nodes.interpolators.BaseInterpolator import FeatureInterpolatorBase
from modules.pose.nodes.Nodes import NodeConfigBase
from typing import cast


class LerpInterpolatorConfig(NodeConfigBase):
    """Configuration for pose linear interpolation with automatic change notification."""

    def __init__(self, input_frequency: float = 30.0) -> None:
        super().__init__()
        self.input_frequency: float = input_frequency


class FeatureLerpInterpolator(FeatureInterpolatorBase[LerpInterpolatorConfig]):
    """Generic pose feature linear interpolator."""

    # Registry mapping feature classes to interpolator classes
    INTERPOLATOR_REGISTRY = {
        Angles: AngleLerp,
        BBox: VectorLerp,
        Points2D: PointLerp,
        Symmetry: VectorLerp,
    }

    def __init__(self, config: LerpInterpolatorConfig, feature_class: type, attr_name: str):
        super().__init__(config, feature_class, attr_name)

    def _create_interpolator(self, interpolator_cls: type, vector_size: int,
                            clamp_range: tuple[float, float] | None):
        """Create the underlying lerp interpolator instance."""
        return interpolator_cls(
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
class AngleLerpInterpolator(FeatureLerpInterpolator):
    def __init__(self, config: LerpInterpolatorConfig) -> None:
        super().__init__(config, Angles, "angles")


class BBoxLerpInterpolator(FeatureLerpInterpolator):
    def __init__(self, config: LerpInterpolatorConfig) -> None:
        super().__init__(config, BBox, "bbox")


class DeltaLerpInterpolator(FeatureLerpInterpolator):
    def __init__(self, config: LerpInterpolatorConfig) -> None:
        super().__init__(config, Angles, "deltas")


class PointLerpInterpolator(FeatureLerpInterpolator):
    def __init__(self, config: LerpInterpolatorConfig) -> None:
        super().__init__(config, Points2D, "points")