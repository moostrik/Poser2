"""Pose smoothing filters using EMA (Exponential Moving Average) with attack/release.

Provides asymmetric smoothing for angles, points, and deltas with different
response rates for increasing vs decreasing values.
"""

# Standard library imports
from collections import defaultdict

# Third-party imports
import numpy as np

# Pose imports
from .._utils.ArrayEmaSmooth import EMASmooth, AngleEMASmooth, PointEMASmooth
from ...features import Angles, Points2D, AngleVelocity, AngleMotion, AngleSymmetry, BBox, Similarity, BaseFeature
from ..Nodes import FilterNode
from ...frame import Frame, replace
from modules.settings import BaseSettings, Field


class EmaSmootherSettings(BaseSettings):
    """Configuration for EMA smoothing with automatic change notification."""
    attack:  Field[float] = Field(0.95)
    release: Field[float] = Field(0.8)


class FeatureEmaSmoother(FilterNode):
    """Generic pose feature smoother using EMA with attack/release."""

    # Registry mapping feature types to smoother classes
    _SMOOTH_MAP: dict[type[BaseFeature], type] = defaultdict(
        lambda: EMASmooth,
        {
            Angles: AngleEMASmooth,
            Points2D: PointEMASmooth,
        }
    )

    def __init__(self, config: EmaSmootherSettings, feature_type: type[BaseFeature]) -> None:
        self._config: EmaSmootherSettings = config
        self._feature_type: type[BaseFeature] = feature_type
        smoother_cls = self._SMOOTH_MAP[feature_type]
        self._smoother = smoother_cls(
            vector_size=feature_type.length(),
            attack=config.attack,
            release=config.release,
            clamp_range=feature_type.range()
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
        feature_data = pose[self._feature_type]
        self._smoother.update(feature_data.values)
        smoothed_values: np.ndarray = self._smoother.value
        smoothed_data = type(feature_data)(values=smoothed_values, scores=feature_data.scores)
        return replace(pose, {self._feature_type: smoothed_data})

    def reset(self) -> None:
        self._smoother.reset()


# Convenience classes
class BBoxEmaSmoother(FeatureEmaSmoother):
    def __init__(self, config: EmaSmootherSettings) -> None:
        super().__init__(config, BBox)


class PointEmaSmoother(FeatureEmaSmoother):
    def __init__(self, config: EmaSmootherSettings) -> None:
        super().__init__(config, Points2D)


class AngleEmaSmoother(FeatureEmaSmoother):
    def __init__(self, config: EmaSmootherSettings) -> None:
        super().__init__(config, Angles)


class AngleVelEmaSmoother(FeatureEmaSmoother):
    def __init__(self, config: EmaSmootherSettings) -> None:
        super().__init__(config, AngleVelocity)


class AngleSymEmaSmoother(FeatureEmaSmoother):
    def __init__(self, config: EmaSmootherSettings) -> None:
        super().__init__(config, AngleSymmetry)


class SimilarityEmaSmoother(FeatureEmaSmoother):
    def __init__(self, config: EmaSmootherSettings) -> None:
        super().__init__(config, Similarity)


class AngleMotionEmaSmoother(FeatureEmaSmoother):
    def __init__(self, config: EmaSmootherSettings) -> None:
        super().__init__(config, AngleMotion)
