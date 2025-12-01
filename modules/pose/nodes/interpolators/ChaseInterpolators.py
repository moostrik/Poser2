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
from dataclasses import replace
from threading import Lock
from time import monotonic
from collections import defaultdict
from typing import cast

# Third-party imports
import numpy as np

# Pose imports
from modules.pose.features import PoseFeatureType, Angles, BBox, Points2D, AngleSymmetry
from modules.pose.nodes._utils.ArrayChase import AngleChase, PointChase, Chase
from modules.pose.nodes.Nodes import InterpolatorNode, NodeConfigBase
from modules.pose.nodes.interpolators.BaseInterpolator import FeatureInterpolatorBase
from modules.pose.Frame import Frame, FrameField


class ChaseInterpolatorConfig(NodeConfigBase):
    """Configuration for pose chase interpolation with automatic change notification."""

    def __init__(self, input_frequency: float = 30.0, responsiveness: float = 0.2, friction: float = 0.03) -> None:
        super().__init__()
        self.input_frequency: float = input_frequency
        self.responsiveness: float = responsiveness
        self.friction: float = friction


class FeatureChaseInterpolator(FeatureInterpolatorBase[ChaseInterpolatorConfig]):
    """Generic pose feature chase interpolator."""

    _INTERP_MAP: defaultdict[FrameField, type] = defaultdict(
        lambda: Chase,
        {
            FrameField.angles: AngleChase,
            FrameField.points: PointChase,
        }
    )


    def _create_interpolator(self):
        """Create the underlying rate limiter instance."""
        interpolator_cls = self._INTERP_MAP[self._pose_field]
        vector_size = len(self._pose_field.get_type().enum())
        clamp_range = self._pose_field.get_type().range()
        return interpolator_cls(
            vector_size=vector_size,
            input_frequency=self._config.input_frequency,
            responsiveness=self._config.responsiveness,
            friction=self._config.friction,
            clamp_range=clamp_range
        )


    def _on_config_changed(self) -> None:
        """Handle configuration changes by updating interpolator parameters."""
        with self._lock:
            cast(Chase, self._interpolator).input_frequency = self._config.input_frequency
            cast(Chase, self._interpolator).responsiveness = self._config.responsiveness
            cast(Chase, self._interpolator).friction = self._config.friction


# Convenience classes
class BBoxChaseInterpolator(FeatureChaseInterpolator):
    def __init__(self, config: ChaseInterpolatorConfig) -> None:
        super().__init__(config, FrameField.bbox)


class PointChaseInterpolator(FeatureChaseInterpolator):
    def __init__(self, config: ChaseInterpolatorConfig) -> None:
        super().__init__(config, FrameField.points)


class AngleChaseInterpolator(FeatureChaseInterpolator):
    def __init__(self, config: ChaseInterpolatorConfig) -> None:
        super().__init__(config, FrameField.angles)


class AngleVelChaseInterpolator(FeatureChaseInterpolator):
    def __init__(self, config: ChaseInterpolatorConfig) -> None:
        super().__init__(config, FrameField.angle_vel)


class AngleSymChaseInterpolator(FeatureChaseInterpolator):
    def __init__(self, config: ChaseInterpolatorConfig) -> None:
        super().__init__(config, FrameField.angle_sym)


class SimilarityChaseInterpolator(FeatureChaseInterpolator):
    def __init__(self, config: ChaseInterpolatorConfig) -> None:
        super().__init__(config, FrameField.similarity)
