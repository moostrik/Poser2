"""Pose predict interpolators: the crossing that shows a feature where its fit puts it at the time it is shown.

``set(pose)`` adds the frame's values and stamp to a fit on the last ``fit_frames`` frames (linear or quadratic,
``method``) and keeps the fit it replaces; ``update(now)`` evaluates the fit at the tick's time plus
``output_delay`` (the time from the tick to the wall or the screen, which the pipeline cannot see), capped at
``max_horizon`` past the newest stamp, with the previous fit eased out by the ``blend_mode`` so a frame never
jumps. With ``step`` FIXED the tick's time is snapped to a grid of ``output_frequency`` intervals, re-anchored
when the clock drifts more than one interval from it, so the horizon advances by exactly one interval per
tick; MEASURED and CAPPED take the time as passed, ``max_horizon`` being the cap. ``horizon`` reads the mean
distance from the newest stamp to the evaluated time over the last second.

Thread safety: the base class's lock serializes ``set()`` (the frame's thread) and ``update()`` (the tick's).
"""

# Standard library imports
from collections import defaultdict
from typing import cast

# Third-party imports
import numpy as np

# Pose imports
from ...features import Angles, AngleVelocity, AngleSymmetry, Azimuth, BBox, Distance, Points2D, Similarity, BaseFeature
from .._utils.array_predict import PredictionMethod
from .._utils.array_predict_blend import AnglePredictBlend, BlendMode, PointPredictBlend, PredictBlend
from .base_interpolator import FeatureInterpolatorBase, InterpolatorSettingsBase, TickStep
from ...frame import Frame, replace
from modules.settings import Field

_HORIZON_PUBLISH: float = 1.0     # seconds between writes of the horizon readout


class PredictInterpolatorSettings(InterpolatorSettingsBase):
    """Configuration for the predict interpolator."""
    method:       Field[PredictionMethod] = Field(PredictionMethod.LINEAR, description="The fit: none, linear or quadratic")
    fit_frames:   Field[int]       = Field(3, min=2, max=10, description="Frames the fit spans")
    output_delay: Field[float]     = Field(0.0, min=0.0, max=0.5, step=0.001, description="Seconds from the tick to the show, aimed past the tick's time")
    blend_mode:   Field[BlendMode] = Field(BlendMode.RAMP, description="How a frame's correction is eased in")
    blend:        Field[float]     = Field(0.0, min=0.0, max=1.0, step=0.001, description="Seconds a correction is spread over (ramp) or its time constant (dampen); 0 = one camera interval")
    max_horizon:  Field[float]     = Field(0.1, min=0.0, max=1.0, step=0.001, description="Farthest the fit is evaluated past the newest frame (s)")
    horizon:      Field[float]     = Field(0.0, access=Field.READ, description="Mean distance from the newest frame to the evaluated time over the last second (ms)")


class FeaturePredictInterpolator(FeatureInterpolatorBase[PredictInterpolatorSettings]):
    """Generic pose feature predict interpolator; see the module docstring."""

    _INTERP_MAP: defaultdict[type[BaseFeature], type] = defaultdict(
        lambda: PredictBlend,
        {
            Angles: AnglePredictBlend,
            Azimuth: AnglePredictBlend,
            Points2D: PointPredictBlend,
        }
    )

    def __init__(self, settings: PredictInterpolatorSettings, feature_type: type[BaseFeature]) -> None:
        # The FIXED grid: anchored at the first tick, counted per tick, re-anchored on drift
        self._grid_anchor: float | None = None
        self._grid_index: int = 0
        # The horizon readout: a running sum over the last second
        self._horizon_sum: float = 0.0
        self._horizon_count: int = 0
        self._horizon_published: float | None = None
        super().__init__(settings, feature_type)

    def _create_interpolator(self):
        """The array predict blend for the feature; the base's dt-stepping protocol does not apply to it."""
        interpolator_cls = self._INTERP_MAP[self._feature_type]
        return interpolator_cls(
            vector_size=self._feature_type.length(),
            method=self.settings.method,
            fit_frames=self.settings.fit_frames,
            blend_mode=self.settings.blend_mode,
            blend=self.settings.blend,
            max_horizon=self.settings.max_horizon,
            clamp_range=self._feature_type.range()
        )

    def _on_settings_changed(self, _=None) -> None:
        previous_interval = self._output_interval
        super()._on_settings_changed()
        with self._lock:
            blend = cast(PredictBlend, self._interpolator)
            blend.fit.method = self.settings.method
            blend.fit.fit_frames = self.settings.fit_frames
            blend.blend_mode = self.settings.blend_mode
            blend.blend = self.settings.blend
            blend.max_horizon = self.settings.max_horizon
            if self._output_interval != previous_interval and self._grid_anchor is not None:
                # The grid goes on from where it is, at the new interval
                self._grid_anchor += self._grid_index * previous_interval
                self._grid_index = 0

    def set(self, pose: Frame) -> None:
        """Add the frame's feature and stamp to the fit. Call at the input rate."""
        feature_data = pose[self._feature_type]
        with self._lock:
            cast(PredictBlend, self._interpolator).set_target(feature_data.values, pose.timestamp)
            self._last_pose = pose

    def update(self, now: float) -> Frame | None:
        """The feature where the fit puts it at the tick's time (by the step mode) plus ``output_delay``."""
        with self._lock:
            if self._last_pose is None:
                return None
            last_pose = self._last_pose
            blend = cast(PredictBlend, self._interpolator)
            time = self._tick_time(now) + self.settings.output_delay
            values: np.ndarray = blend.at(time)
            horizon = self._record_horizon(blend.horizon, now)
        if horizon is not None:
            self.settings.horizon = horizon          # outside the lock: the write calls the settings listeners

        feature_data = last_pose[self._feature_type]
        interpolated_data = self._create_interpolated_data(feature_data, values)
        return replace(last_pose, {self._feature_type: interpolated_data})

    def reset(self) -> None:
        super().reset()
        with self._lock:
            self._grid_anchor = None
            self._grid_index = 0

    def _tick_time(self, now: float) -> float:
        """The time the tick is for: ``now`` as passed, or with FIXED its place on the grid."""
        if self._step != TickStep.FIXED:
            return now
        if self._grid_anchor is None:
            self._grid_anchor, self._grid_index = now, 0
            return now
        self._grid_index += 1
        grid = self._grid_anchor + self._grid_index * self._output_interval
        if abs(now - grid) > self._output_interval:
            self._grid_anchor, self._grid_index = now, 0     # the clock drifted a whole interval: re-anchor
            return now
        return grid

    def _record_horizon(self, horizon: float, now: float) -> float | None:
        """Add a tick's horizon to the running mean; the mean in ms when a second has passed, else None."""
        self._horizon_sum += horizon
        self._horizon_count += 1
        if self._horizon_published is None:
            self._horizon_published = now
            return None
        if now - self._horizon_published < _HORIZON_PUBLISH:
            return None
        mean = 1000.0 * self._horizon_sum / self._horizon_count
        self._horizon_sum, self._horizon_count, self._horizon_published = 0.0, 0, now
        return mean


# Convenience classes
class BBoxPredictInterpolator(FeaturePredictInterpolator):
    def __init__(self, settings: PredictInterpolatorSettings) -> None:
        super().__init__(settings, BBox)


class PointPredictInterpolator(FeaturePredictInterpolator):
    def __init__(self, settings: PredictInterpolatorSettings) -> None:
        super().__init__(settings, Points2D)


class AnglePredictInterpolator(FeaturePredictInterpolator):
    def __init__(self, settings: PredictInterpolatorSettings) -> None:
        super().__init__(settings, Angles)


class AngleVelocityPredictInterpolator(FeaturePredictInterpolator):
    def __init__(self, settings: PredictInterpolatorSettings) -> None:
        super().__init__(settings, AngleVelocity)


class AngleSymmetryPredictInterpolator(FeaturePredictInterpolator):
    def __init__(self, settings: PredictInterpolatorSettings) -> None:
        super().__init__(settings, AngleSymmetry)


class SimilarityPredictInterpolator(FeaturePredictInterpolator):
    def __init__(self, settings: PredictInterpolatorSettings) -> None:
        super().__init__(settings, Similarity)


class AzimuthPredictInterpolator(FeaturePredictInterpolator):
    def __init__(self, settings: PredictInterpolatorSettings) -> None:
        super().__init__(settings, Azimuth)


class DistancePredictInterpolator(FeaturePredictInterpolator):
    def __init__(self, settings: PredictInterpolatorSettings) -> None:
        super().__init__(settings, Distance)
