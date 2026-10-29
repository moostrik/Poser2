"""Pose chase interpolation filters for smooth motion interpolation.

Provides perpetual chase interpolation for angles, points, and deltas with
proper handling of circular values and coordinate clamping. Uses dual-frequency
architecture: process() at input rate, update() at render rate.

Thread Safety:
--------------
Designed for multi-threaded operation:
- Input thread: Calls process() at ~30 FPS
- Render thread: Calls update() at 60+ FPS

VectorChaseInterpolator classes are NOT thread-safe. FeatureChaseInterpolator
serializes all access using a lock to prevent concurrent set_target() and update() calls
and makes sure the last_pose always corresponds to interpolator's last set target.
"""

# Standard library imports
from collections import defaultdict
from typing import cast

# Third-party imports
import numpy as np

# Pose imports
from ...features import Angles, AngleVelocity, AngleSymmetry, Azimuth, BBox, Distance, Points2D, Similarity, BaseFeature
from .._utils.array_chase import AngleChase, PointChase, Chase
from .base_interpolator import FeatureInterpolatorBase, InterpolatorSettingsBase
from ...frame import Frame
from modules.settings import Field


class ChaseInterpolatorSettings(InterpolatorSettingsBase):
    """Configuration for pose chase interpolation."""
    responsiveness: Field[float] = Field(0.2)
    friction:       Field[float] = Field(0.03)


class FeatureChaseInterpolator(FeatureInterpolatorBase[ChaseInterpolatorSettings]):
    """Generic pose feature chase interpolator."""

    _INTERP_MAP: defaultdict[type[BaseFeature], type] = defaultdict(
        lambda: Chase,
        {
            Angles: AngleChase,
            Azimuth: AngleChase,
            Points2D: PointChase,
        }
    )


    def _create_interpolator(self):
        """Create the underlying chase interpolator instance."""
        interpolator_cls = self._INTERP_MAP[self._feature_type]
        vector_size = self._feature_type.length()
        clamp_range = self._feature_type.range()
        return interpolator_cls(
            vector_size=vector_size,
            input_frequency=self.settings.input_frequency,
            responsiveness=self.settings.responsiveness,
            friction=self.settings.friction,
            clamp_range=clamp_range
        )


    def _on_settings_changed(self, _=None) -> None:
        """Handle configuration changes by updating interpolator parameters."""
        super()._on_settings_changed()
        with self._lock:
            cast(Chase, self._interpolator).input_frequency = self.settings.input_frequency
            cast(Chase, self._interpolator).responsiveness = self.settings.responsiveness
            cast(Chase, self._interpolator).friction = self.settings.friction


# Convenience classes
class BBoxChaseInterpolator(FeatureChaseInterpolator):
    def __init__(self, settings:ChaseInterpolatorSettings) -> None:
        super().__init__(settings,BBox)


class PointChaseInterpolator(FeatureChaseInterpolator):
    def __init__(self, settings:ChaseInterpolatorSettings) -> None:
        super().__init__(settings,Points2D)


class AngleChaseInterpolator(FeatureChaseInterpolator):
    def __init__(self, settings:ChaseInterpolatorSettings) -> None:
        super().__init__(settings,Angles)


class AngleVelocityChaseInterpolator(FeatureChaseInterpolator):
    def __init__(self, settings:ChaseInterpolatorSettings) -> None:
        super().__init__(settings,AngleVelocity)


class AngleSymmetryChaseInterpolator(FeatureChaseInterpolator):
    def __init__(self, settings:ChaseInterpolatorSettings) -> None:
        super().__init__(settings,AngleSymmetry)


class SimilarityChaseInterpolator(FeatureChaseInterpolator):
    def __init__(self, settings:ChaseInterpolatorSettings) -> None:
        super().__init__(settings,Similarity)


class AzimuthChaseInterpolator(FeatureChaseInterpolator):
    def __init__(self, settings:ChaseInterpolatorSettings) -> None:
        super().__init__(settings,Azimuth)


class DistanceChaseInterpolator(FeatureChaseInterpolator):
    def __init__(self, settings:ChaseInterpolatorSettings) -> None:
        super().__init__(settings,Distance)
