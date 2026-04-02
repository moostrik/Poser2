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
from modules.pose.nodes.Nodes import FilterNode
from modules.pose.frame import Frame, FrameField
from modules.settings import Settings, Field


class EmaSmootherSettings(Settings):
    """Configuration for EMA smoothing with automatic change notification."""
    attack:  Field[float] = Field(0.95)
    release: Field[float] = Field(0.8)


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

    def __init__(self, config: EmaSmootherSettings, pose_field: FrameField) -> None:
        self._config: EmaSmootherSettings = config
        self._pose_field: FrameField = pose_field
        smoother_cls = self._SMOOTH_MAP[pose_field]
        self._smoother = smoother_cls(
            vector_size=len(pose_field.get_type().enum()),
            attack=config.attack,
            release=config.release,
            clamp_range=pose_field.get_type().range()
        )
        self._config.bind_all(self._on_config_changed)

    def __del__(self):
        """Cleanup config listener to prevent memory leaks."""
        try:
            self._config.unbind_all(self._on_config_changed)
        except (AttributeError, ValueError):
            pass  # Config already cleaned up or listener not found

    def _on_config_changed(self, _=None) -> None:
        self._smoother.attack = self._config.attack
        self._smoother.release = self._config.release

    @property
    def config(self) -> EmaSmootherSettings:
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
    def __init__(self, config: EmaSmootherSettings) -> None:
        super().__init__(config, FrameField.bbox)


class PointEmaSmoother(FeatureEmaSmoother):
    def __init__(self, config: EmaSmootherSettings) -> None:
        super().__init__(config, FrameField.points)


class AngleEmaSmoother(FeatureEmaSmoother):
    def __init__(self, config: EmaSmootherSettings) -> None:
        super().__init__(config, FrameField.angles)


class AngleVelEmaSmoother(FeatureEmaSmoother):
    def __init__(self, config: EmaSmootherSettings) -> None:
        super().__init__(config, FrameField.angle_vel)


class AngleSymEmaSmoother(FeatureEmaSmoother):
    def __init__(self, config: EmaSmootherSettings) -> None:
        super().__init__(config, FrameField.angle_sym)


class SimilarityEmaSmoother(FeatureEmaSmoother):
    def __init__(self, config: EmaSmootherSettings) -> None:
        super().__init__(config, FrameField.similarity)


class AngleMotionEmaSmoother(FeatureEmaSmoother):
    def __init__(self, config: EmaSmootherSettings) -> None:
        super().__init__(config, FrameField.angle_motion)
