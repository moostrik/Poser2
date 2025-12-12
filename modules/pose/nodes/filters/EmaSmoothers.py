"""Pose smoothing filters using EMA (Exponential Moving Average) with attack/release.

Provides asymmetric smoothing for angles, points, and deltas with different
response rates for increasing vs decreasing values.
"""

# Standard library imports
from dataclasses import replace
from collections import defaultdict

# Third-party imports
import numpy as np

# Pose imports
from modules.pose.nodes._utils.ArrayEmaSmooth import EMASmooth, AngleEMASmooth, PointEMASmooth
from modules.pose.nodes.Nodes import FilterNode, NodeConfigBase
from modules.pose.Frame import Frame, FrameField


class EmaSmootherConfig(NodeConfigBase):
    """Configuration for EMA smoothing with automatic change notification."""

    def __init__(self, attack: float = 0.95, release: float = 0.8) -> None:
        super().__init__()
        self.attack: float = attack
        self.release: float = release


class FeatureEmaSmoother(FilterNode):
    """Generic pose feature smoother using EMA with attack/release."""

    # Registry mapping feature classes to smoother classes
    _SMOOTH_MAP = defaultdict(
        lambda: EMASmooth,
        {
            FrameField.angles: AngleEMASmooth,
            FrameField.points: PointEMASmooth,
        }
    )

    def __init__(self, config: EmaSmootherConfig, pose_field: FrameField) -> None:
        self._config: EmaSmootherConfig = config
        self._pose_field: FrameField = pose_field
        smoother_cls = self._SMOOTH_MAP[pose_field]
        self._smoother = smoother_cls(
            vector_size=len(pose_field.get_type().enum()),
            attack=config.attack,
            release=config.release,
            clamp_range=pose_field.get_type().range()
        )
        self._config.add_listener(self._on_config_changed)

    def __del__(self):
        """Cleanup config listener to prevent memory leaks."""
        try:
            self._config.remove_listener(self._on_config_changed)
        except (AttributeError, ValueError):
            pass  # Config already cleaned up or listener not found

    def _on_config_changed(self) -> None:
        self._smoother.attack = self._config.attack
        self._smoother.release = self._config.release
        print(f"[EmaSmoother] Updated parameters: attack={self._smoother.attack:.4f}, release={self._smoother.release:.4f}")

    @property
    def config(self) -> EmaSmootherConfig:
        return self._config

    def process(self, pose: Frame) -> Frame:
        feature_data = pose.get_feature(self._pose_field)
        self._smoother.update(feature_data.values)
        smoothed_values: np.ndarray = self._smoother.value
        smoothed_data = type(feature_data)(values=smoothed_values, scores=feature_data.scores)
        return replace(pose, **{self._pose_field.name: smoothed_data})

    def reset(self) -> None:
        self._smoother.reset()


# Convenience classes
class BBoxEmaSmoother(FeatureEmaSmoother):
    def __init__(self, config: EmaSmootherConfig) -> None:
        super().__init__(config, FrameField.bbox)


class PointEmaSmoother(FeatureEmaSmoother):
    def __init__(self, config: EmaSmootherConfig) -> None:
        super().__init__(config, FrameField.points)


class AngleEmaSmoother(FeatureEmaSmoother):
    def __init__(self, config: EmaSmootherConfig) -> None:
        super().__init__(config, FrameField.angles)


class AngleVelEmaSmoother(FeatureEmaSmoother):
    def __init__(self, config: EmaSmootherConfig) -> None:
        super().__init__(config, FrameField.angle_vel)


class AngleSymEmaSmoother(FeatureEmaSmoother):
    def __init__(self, config: EmaSmootherConfig) -> None:
        super().__init__(config, FrameField.angle_sym)


class SimilarityEmaSmoother(FeatureEmaSmoother):
    def __init__(self, config: EmaSmootherConfig) -> None:
        super().__init__(config, FrameField.similarity)


class AngleMotionEmaSmoother(FeatureEmaSmoother):
    def __init__(self, config: EmaSmootherConfig) -> None:
        super().__init__(config, FrameField.angle_motion)
