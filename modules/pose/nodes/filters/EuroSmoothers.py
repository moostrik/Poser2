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
from ...features import Angles, Points2D, AngleVelocity, AngleSymmetry, Azimuth, BBox, Similarity, BaseFeature
from ..Nodes import FilterNode
from ...frame import Frame, replace
from modules.settings import BaseSettings, Field, Widget


class EuroSmootherSettings(BaseSettings):
    """Configuration for the One Euro smoother: two low-pass filters and one rule.

        difference = (raw - previous smoothed) * frame rate
        difference = low-pass(difference, d_cutoff)      # signed: jitter cancels, a move stays
        cutoff     = min_cutoff + cutoff_rise * |difference|
        smoothed   = low-pass(raw, cutoff)

    One frame of a point at 30 fps, min_cutoff 1.5, cutoff_rise 5: raw 0.52, smoothed 0.50
    -> difference 0.02 -> * 30 = 0.6 -> low-passed by d_cutoff -> * 5 = +3 Hz -> cutoff 4.5 Hz.

    The difference is in whatever the smoothed values are counted in (point, bbox: image
    size 0..1; angle, azimuth: sin/cos of the angle; angle velocity: rad/s; similarity,
    angle symmetry: score), so cutoff_rise values are not comparable between smoothers.

    Tuning: stand still and lower min_cutoff until the jitter is gone, then move fast and
    raise cutoff_rise until it stops trailing. d_cutoff 1.0 is the One Euro default and
    measured near-optimal. min_cutoff is frame-rate independent; check cutoff_rise after
    changing the frame rate.
    """
    frequency:   Field[float] = Field(30.0,  access=Field.INIT)
    min_cutoff:  Field[float] = Field(1.0,   min=0.001, max=10.0,  widget=Widget.log_slider, description="Minimum low-pass cutoff (Hz). Lower = more smooth, more lag")
    cutoff_rise: Field[float] = Field(0.025, min=0.001, max=100.0, widget=Widget.log_slider, description="Cutoff rise (Hz) = this × difference between raw and smoothed per second. Lower = more smooth, more lag")
    d_cutoff:    Field[float] = Field(1.0,   min=0.1,   max=10.0,  visible=False,            description="Low-pass on the difference between raw and smoothed (Hz). Lower = more smooth when still, slower to notice a move")


class FeatureEuroSmoother(FilterNode):
    """Generic pose feature smoother using OneEuroFilter."""

    # Registry mapping feature types to smoother classes
    _SMOOTH_MAP: dict[type[BaseFeature], type] = defaultdict(
        lambda: EuroSmooth,
        {
            Angles: AngleEuroSmooth,
            Azimuth: AngleEuroSmooth,
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
            cutoff_rise=config.cutoff_rise,
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
        self._smoother.cutoff_rise = self._config.cutoff_rise
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


class AzimuthEuroSmoother(FeatureEuroSmoother):
    def __init__(self, config: EuroSmootherSettings) -> None:
        super().__init__(config, Azimuth)