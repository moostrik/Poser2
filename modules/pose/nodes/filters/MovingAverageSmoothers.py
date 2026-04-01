"""Pose smoothing filters using Moving Average with configurable window types.

Provides predictable smoothing with bounded history window and multiple
weighting options: uniform, triangular, gaussian, exponential.
"""

# Standard library imports
from dataclasses import replace
from collections import defaultdict

# Third-party imports
import numpy as np

# Pose imports
from modules.pose.nodes._utils.ArrayMovingAverage import MovingAverage, WindowType
from modules.pose.nodes.Nodes import FilterNode
from modules.pose.Frame import Frame, FrameField
from modules.settings import Settings, Field


# Re-export WindowType for convenience
__all__ = ['WindowType', 'MovingAverageSettings', 'FeatureMovingAverageSmoother',
           'AngleMotionMovingAverageSmoother', 'SimilarityMovingAverageSmoother']


class MovingAverageSettings(Settings):
    """Configuration for moving average smoothing."""
    window_size: Field[int]        = Field(30)
    window_type: Field[WindowType] = Field(WindowType.TRIANGULAR)


class FeatureMovingAverageSmoother(FilterNode):
    """Generic pose feature smoother using weighted moving average."""

    def __init__(self, config: MovingAverageSettings, pose_field: FrameField) -> None:
        self._config: MovingAverageSettings = config
        self._pose_field: FrameField = pose_field
        self._smoother = MovingAverage(
            vector_size=len(pose_field.get_type().enum()),
            window_size=config.window_size,
            window_type=config.window_type,
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
        self._smoother.window_size = self._config.window_size
        self._smoother.window_type = self._config.window_type

    @property
    def config(self) -> MovingAverageSettings:
        return self._config

    def process(self, pose: Frame) -> Frame:
        feature_data = pose.get_feature(self._pose_field)
        self._smoother.update(feature_data.values)
        smoothed_values: np.ndarray = self._smoother.value
        smoothed_data = type(feature_data)(values=smoothed_values, scores=feature_data.scores)
        return replace(pose, **{self._pose_field.name: smoothed_data})

    def reset(self) -> None:
        self._smoother.reset()


# Convenience classes for specific features
class AngleMotionMovingAverageSmoother(FeatureMovingAverageSmoother):
    """Moving average smoother for angle motion feature."""
    def __init__(self, config: MovingAverageSettings) -> None:
        super().__init__(config, FrameField.angle_motion)


class SimilarityMovingAverageSmoother(FeatureMovingAverageSmoother):
    """Moving average smoother for similarity feature."""
    def __init__(self, config: MovingAverageSettings) -> None:
        super().__init__(config, FrameField.similarity)


class BBoxMovingAverageSmoother(FeatureMovingAverageSmoother):
    """Moving average smoother for bounding box."""
    def __init__(self, config: MovingAverageSettings) -> None:
        super().__init__(config, FrameField.bbox)


class PointMovingAverageSmoother(FeatureMovingAverageSmoother):
    """Moving average smoother for points."""
    def __init__(self, config: MovingAverageSettings) -> None:
        super().__init__(config, FrameField.points)


class AngleMovingAverageSmoother(FeatureMovingAverageSmoother):
    """Moving average smoother for angles."""
    def __init__(self, config: MovingAverageSettings) -> None:
        super().__init__(config, FrameField.angles)


class AngleVelMovingAverageSmoother(FeatureMovingAverageSmoother):
    """Moving average smoother for angle velocity."""
    def __init__(self, config: MovingAverageSettings) -> None:
        super().__init__(config, FrameField.angle_vel)
