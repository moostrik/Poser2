"""Pose smoothing filters using OneEuroFilter for noise reduction.

Provides smoothing for angles, points, and deltas with proper handling
of circular values and coordinate clamping.
"""

# Standard library imports
from dataclasses import replace
from collections import defaultdict

# Third-party imports
import numpy as np

# Pose imports
from modules.pose.features import PoseFeatureType
from modules.pose.nodes._utils.VectorSmooth import VectorSmooth, AngleSmooth, PointSmooth
from modules.pose.nodes.Nodes import FilterNode, NodeConfigBase
from modules.pose.Frame import Frame, FrameField


class EuroSmootherConfig(NodeConfigBase):
    """Configuration for pose smoothing with automatic change notification."""

    def __init__(self, frequency: float = 30.0, min_cutoff: float = 1.0, beta: float = 0.025, d_cutoff: float = 1.0) -> None:
        super().__init__()
        self.frequency: float = frequency
        self.min_cutoff: float = min_cutoff
        self.beta: float = beta
        self.d_cutoff: float = d_cutoff


class FeatureEuroSmoother(FilterNode):
    """Generic pose feature smoother using OneEuroFilter."""

    # Registry mapping feature classes to smoother classes
    _SMOOTH_MAP = defaultdict(
        lambda: VectorSmooth,
        {
            FrameField.angles: AngleSmooth,
            FrameField.points: PointSmooth,
        }
    )

    def __init__(self, config: EuroSmootherConfig, pose_field: FrameField) -> None:
        self._config: EuroSmootherConfig = config
        self._pose_field: FrameField = pose_field
        smoother_cls = self._SMOOTH_MAP[pose_field]
        self._smoother = smoother_cls(
            vector_size=len(pose_field.get_type().feature_enum()),
            frequency=config.frequency,
            min_cutoff=config.min_cutoff,
            beta=config.beta,
            d_cutoff=config.d_cutoff,
            clamp_range=pose_field.get_type().default_range()
        )
        self._config.add_listener(self._on_config_changed)

    def __del__(self):
        """Cleanup config listener to prevent memory leaks."""
        try:
            self._config.remove_listener(self._on_config_changed)
        except (AttributeError, ValueError):
            pass  # Config already cleaned up or listener not found

    def _on_config_changed(self) -> None:
        self._smoother.frequency = self._config.frequency
        self._smoother.min_cutoff = self._config.min_cutoff
        self._smoother.beta = self._config.beta
        self._smoother.d_cutoff = self._config.d_cutoff

    @property
    def config(self) -> EuroSmootherConfig:
        return self._config

    def process(self, pose: Frame) -> Frame:
        feature_data = pose.get_feature(self._pose_field)
        self._smoother.add_sample(feature_data.values)
        smoothed_values: np.ndarray = self._smoother.value
        smoothed_data = type(feature_data)(values=smoothed_values, scores=feature_data.scores)
        return replace(pose, **{self._pose_field.name: smoothed_data})

    def reset(self) -> None:
        self._smoother.reset()


# Convenience classes
class BBoxEuroSmoother(FeatureEuroSmoother):
    def __init__(self, config: EuroSmootherConfig) -> None:
        super().__init__(config, FrameField.bbox)


class PointEuroSmoother(FeatureEuroSmoother):
    def __init__(self, config: EuroSmootherConfig) -> None:
        super().__init__(config, FrameField.points)


class AngleEuroSmoother(FeatureEuroSmoother):
    def __init__(self, config: EuroSmootherConfig) -> None:
        super().__init__(config, FrameField.angles)


class AngleVelEuroSmoother(FeatureEuroSmoother):
    def __init__(self, config: EuroSmootherConfig) -> None:
        super().__init__(config, FrameField.angle_vel)


class AngleSymEuroSmoother(FeatureEuroSmoother):
    def __init__(self, config: EuroSmootherConfig) -> None:
        super().__init__(config, FrameField.angle_sym)