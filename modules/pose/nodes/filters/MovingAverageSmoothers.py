"""Pose smoothing filters using Moving Average with configurable window types.

Provides predictable smoothing with bounded history window and multiple
weighting options: uniform, triangular, gaussian, exponential.
"""

# Standard library imports
from collections import defaultdict

# Third-party imports
import numpy as np

# Pose imports
from modules.pose.nodes._utils.ArrayMovingAverage import MovingAverage, WindowType
from modules.pose.features import Angles, AngleVelocity, AngleMotion, BBox, Points2D, Similarity
from modules.pose.features.base import BaseFeature
from modules.pose.nodes.Nodes import FilterNode
from modules.pose.frame import Frame, replace
from modules.settings import BaseSettings, Field


# Re-export WindowType for convenience
__all__ = ['WindowType', 'MovingAverageSettings', 'FeatureMovingAverageSmoother',
           'AngleMotionMovingAverageSmoother', 'SimilarityMovingAverageSmoother']


class MovingAverageSettings(BaseSettings):
    """Configuration for moving average smoothing."""
    window_size: Field[int]        = Field(30)
    window_type: Field[WindowType] = Field(WindowType.TRIANGULAR)


class FeatureMovingAverageSmoother(FilterNode):
    """Generic pose feature smoother using weighted moving average."""

    def __init__(self, config: MovingAverageSettings, feature_type: type[BaseFeature]) -> None:
        self._config: MovingAverageSettings = config
        self._feature_type: type[BaseFeature] = feature_type
        self._smoother = MovingAverage(
            vector_size=feature_type.length(),
            window_size=config.window_size,
            window_type=config.window_type,
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
        self._smoother.window_size = self._config.window_size
        self._smoother.window_type = self._config.window_type

    @property
    def config(self) -> MovingAverageSettings:
        return self._config

    def process(self, pose: Frame) -> Frame:
        feature_data = pose[self._feature_type]
        self._smoother.update(feature_data.values)
        smoothed_values: np.ndarray = self._smoother.value
        smoothed_data = type(feature_data)(values=smoothed_values, scores=feature_data.scores)
        return replace(pose, {self._feature_type: smoothed_data})

    def reset(self) -> None:
        self._smoother.reset()


# Convenience classes for specific features
class AngleMotionMovingAverageSmoother(FeatureMovingAverageSmoother):
    """Moving average smoother for angle motion feature."""
    def __init__(self, config: MovingAverageSettings) -> None:
        super().__init__(config, AngleMotion)


class SimilarityMovingAverageSmoother(FeatureMovingAverageSmoother):
    """Moving average smoother for similarity feature."""
    def __init__(self, config: MovingAverageSettings) -> None:
        super().__init__(config, Similarity)


class BBoxMovingAverageSmoother(FeatureMovingAverageSmoother):
    """Moving average smoother for bounding box."""
    def __init__(self, config: MovingAverageSettings) -> None:
        super().__init__(config, BBox)


class PointMovingAverageSmoother(FeatureMovingAverageSmoother):
    """Moving average smoother for points."""
    def __init__(self, config: MovingAverageSettings) -> None:
        super().__init__(config, Points2D)


class AngleMovingAverageSmoother(FeatureMovingAverageSmoother):
    """Moving average smoother for angles."""
    def __init__(self, config: MovingAverageSettings) -> None:
        super().__init__(config, Angles)


class AngleVelMovingAverageSmoother(FeatureMovingAverageSmoother):
    """Moving average smoother for angle velocity."""
    def __init__(self, config: MovingAverageSettings) -> None:
        super().__init__(config, AngleVelocity)
