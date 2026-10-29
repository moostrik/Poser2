"""Pose smoothing filters using EMA (Exponential Moving Average) with attack/release.

Provides asymmetric smoothing for angles, points, and deltas with different
response rates for increasing vs decreasing values.
"""

# Standard library imports
from collections import defaultdict

# Third-party imports
import numpy as np

# Pose imports
from .._utils.array_ema_smooth import EmaSmooth, AngleEmaSmooth, PointEmaSmooth
from ...features import Angles, Points2D, AngleVelocity, AngleMotion, AngleSymmetry, BBox, Similarity, BaseFeature
from ..nodes import FilterNode
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
        lambda: EmaSmooth,
        {
            Angles: AngleEmaSmooth,
            Points2D: PointEmaSmooth,
        }
    )

    def __init__(self, settings: EmaSmootherSettings, feature_type: type[BaseFeature]) -> None:
        self.settings: EmaSmootherSettings = settings
        self._feature_type: type[BaseFeature] = feature_type
        smoother_cls = self._SMOOTH_MAP[feature_type]
        self._smoother = smoother_cls(
            vector_size=feature_type.length(),
            attack=settings.attack,
            release=settings.release,
            clamp_range=feature_type.range()
        )
        self.settings.bind_all(self._on_settings_changed)

    def __del__(self):
        """Cleanup settings listener to prevent memory leaks."""
        try:
            self.settings.unbind_all(self._on_settings_changed)
        except (AttributeError, ValueError):
            pass  # Settings already cleaned up or listener not found

    def _on_settings_changed(self, _=None) -> None:
        self._smoother.attack = self.settings.attack
        self._smoother.release = self.settings.release

    def process(self, pose: Frame) -> Frame:
        feature_data = pose[self._feature_type]
        self._smoother.update(feature_data.values, pose.timestamp)
        smoothed_values: np.ndarray = self._smoother.value
        smoothed_data = type(feature_data)(values=smoothed_values, scores=feature_data.scores)
        return replace(pose, {self._feature_type: smoothed_data})

    def reset(self) -> None:
        self._smoother.reset()


# Convenience classes
class BBoxEmaSmoother(FeatureEmaSmoother):
    def __init__(self, settings:EmaSmootherSettings) -> None:
        super().__init__(settings,BBox)


class PointEmaSmoother(FeatureEmaSmoother):
    def __init__(self, settings:EmaSmootherSettings) -> None:
        super().__init__(settings,Points2D)


class AngleEmaSmoother(FeatureEmaSmoother):
    def __init__(self, settings:EmaSmootherSettings) -> None:
        super().__init__(settings,Angles)


class AngleVelocityEmaSmoother(FeatureEmaSmoother):
    def __init__(self, settings:EmaSmootherSettings) -> None:
        super().__init__(settings,AngleVelocity)


class AngleSymmetryEmaSmoother(FeatureEmaSmoother):
    def __init__(self, settings:EmaSmootherSettings) -> None:
        super().__init__(settings,AngleSymmetry)


class SimilarityEmaSmoother(FeatureEmaSmoother):
    def __init__(self, settings:EmaSmootherSettings) -> None:
        super().__init__(settings,Similarity)


class AngleMotionEmaSmoother(FeatureEmaSmoother):
    def __init__(self, settings:EmaSmootherSettings) -> None:
        super().__init__(settings,AngleMotion)
