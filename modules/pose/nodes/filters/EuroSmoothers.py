"""Pose smoothing filters using OneEuroFilter for noise reduction.

Provides smoothing for angles, points, and deltas with proper handling
of circular values and coordinate clamping.
"""

# Standard library imports
from collections import defaultdict

# Third-party imports
import numpy as np

# Pose imports
from .._utils.ArrayEuroSmooth import EuroSmooth, AngleEuroSmooth, PointEuroSmooth
from ...features import Angles, Points2D, AngleVelocity, AngleSymmetry, BBox, Similarity, BaseFeature
from ..Nodes import FilterNode
from ...frame import Frame, replace
from modules.settings import BaseSettings, Field


class EuroSmootherSettings(BaseSettings):
    """Configuration for pose smoothing."""
    frequency:  Field[float] = Field(30.0,  access=Field.INIT)
    min_cutoff: Field[float] = Field(1.0)
    beta:       Field[float] = Field(0.025)
    d_cutoff:   Field[float] = Field(1.0)


class FeatureEuroSmoother(FilterNode):
    """Generic pose feature smoother using OneEuroFilter."""

    # Registry mapping feature types to smoother classes
    _SMOOTH_MAP: dict[type[BaseFeature], type] = defaultdict(
        lambda: EuroSmooth,
        {
            Angles: AngleEuroSmooth,
            Points2D: PointEuroSmooth,
        }
    )

    def __init__(self, config: EuroSmootherSettings, feature_type: type[BaseFeature]) -> None:
        self._config: EuroSmootherSettings = config
        self._feature_type: type[BaseFeature] = feature_type
        smoother_cls = self._SMOOTH_MAP[feature_type]
        self._smoother = smoother_cls(
            vector_size=feature_type.length(),
            frequency=config.frequency,
            min_cutoff=config.min_cutoff,
            beta=config.beta,
            d_cutoff=config.d_cutoff,
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
        self._smoother.frequency = self._config.frequency
        self._smoother.min_cutoff = self._config.min_cutoff
        self._smoother.beta = self._config.beta
        self._smoother.d_cutoff = self._config.d_cutoff

    @property
    def config(self) -> EuroSmootherSettings:
        return self._config

    def process(self, pose: Frame) -> Frame:
        feature_data = pose[self._feature_type]
        self._smoother.add_sample(feature_data.values)
        smoothed_values: np.ndarray = self._smoother.value
        smoothed_data = type(feature_data)(values=smoothed_values, scores=feature_data.scores)
        return replace(pose, {self._feature_type: smoothed_data})

    def reset(self) -> None:
        self._smoother.reset()


# Convenience classes
class BBoxEuroSmoother(FeatureEuroSmoother):
    def __init__(self, config: EuroSmootherSettings) -> None:
        super().__init__(config, BBox)


class PointEuroSmoother(FeatureEuroSmoother):
    def __init__(self, config: EuroSmootherSettings) -> None:
        super().__init__(config, Points2D)


class AngleEuroSmoother(FeatureEuroSmoother):
    def __init__(self, config: EuroSmootherSettings) -> None:
        super().__init__(config, Angles)


class AngleVelEuroSmoother(FeatureEuroSmoother):
    def __init__(self, config: EuroSmootherSettings) -> None:
        super().__init__(config, AngleVelocity)


class AngleSymEuroSmoother(FeatureEuroSmoother):
    def __init__(self, config: EuroSmootherSettings) -> None:
        super().__init__(config, AngleSymmetry)


class SimilarityEuroSmoother(FeatureEuroSmoother):
    def __init__(self, config: EuroSmootherSettings) -> None:
        super().__init__(config, Similarity)