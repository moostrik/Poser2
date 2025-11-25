"""Pose prediction filters for extrapolating future pose states.

Provides prediction for angles, points, and deltas using linear or quadratic
extrapolation with proper handling of circular values and coordinate clamping.
"""

# Standard library imports
from dataclasses import replace
from collections import defaultdict

# Third-party imports
import numpy as np

# Pose imports
from modules.pose.features import Angles, BBox, Points2D, AngleSymmetry
from modules.pose.nodes._utils.VectorPredict import AnglePredict, PointPredict, VectorPredict, PredictionMethod
from modules.pose.nodes.Nodes import FilterNode, NodeConfigBase
from modules.pose.Frame import Frame, FrameField
from modules.pose.nodes.filters.RateLimiters import RateLimiterConfig


class PredictorConfig(NodeConfigBase):
    """Configuration for pose prediction with automatic change notification."""

    def __init__(self, frequency: float = 30.0, method: PredictionMethod = PredictionMethod.QUADRATIC) -> None:
        super().__init__()
        self.frequency: float = frequency
        self.method: PredictionMethod = method


class FeaturePredictor(FilterNode):
    """Generic pose feature predictor."""

    _PREDICT_MAP = defaultdict(
        lambda: VectorPredict,
        {
            FrameField.angles: AnglePredict,
            FrameField.points: PointPredict,
        }
    )

    def __init__(self, config: PredictorConfig, pose_field: FrameField) -> None:
        self._config = config
        self._pose_field = pose_field
        predictor_cls = self._PREDICT_MAP[pose_field]
        self._predictor = predictor_cls(
            vector_size=len(pose_field.get_type().feature_enum()),
            input_frequency=config.frequency,
            method=config.method,
            clamp_range=pose_field.get_type().default_range()
        )
        self._config.add_listener(self._on_config_changed)

    def __del__(self):
        """Cleanup config listener to prevent memory leaks."""
        try:
            self._config.remove_listener(self._on_config_changed)
        except (AttributeError, ValueError):
            pass

    @property
    def config(self) -> PredictorConfig:
        return self._config

    def _create_predicted_data(self, original_data, predicted_values: np.ndarray):
        """Create feature data with predicted values and adjusted scores.

        Sets scores to 0 where predictions are NaN, preserves original scores otherwise.
        For 2D data (points), checks if ANY coordinate is NaN per element.
        """
        # Check for NaN - handle both 1D and 2D cases
        if predicted_values.ndim > 1:
            has_nan = np.any(np.isnan(predicted_values), axis=-1)
        else:
            has_nan = np.isnan(predicted_values)

        interpolated_scores = np.where(has_nan, 0.0, original_data.scores).astype(np.float32)
        return type(original_data)(values=predicted_values, scores=interpolated_scores)

    def process(self, pose: Frame) -> Frame:
        """Add current feature data to predictor and return pose with predicted values."""
        feature_data = pose.get_feature(self._pose_field)

        # Add sample and get prediction
        self._predictor.add_sample(feature_data.values)
        predicted_values: np.ndarray = self._predictor.value

        # Create new feature data with predicted values
        predicted_data = self._create_predicted_data(feature_data, predicted_values)

        # Return new pose with predicted feature
        return replace(pose, **{self._pose_field.name: predicted_data})

    def reset(self) -> None:
        """Reset the predictor's internal state (clear sample history)."""
        self._predictor.reset()

    def _on_config_changed(self) -> None:
        """Handle configuration changes by updating predictor parameters."""
        self._predictor.input_frequency = self._config.frequency
        self._predictor.method = self._config.method


# Convenience classes

class BBoxPredictor(FeaturePredictor):
    def __init__(self, config: PredictorConfig) -> None:
        super().__init__(config, FrameField.bbox)


class PointPredictor(FeaturePredictor):
    def __init__(self, config: PredictorConfig) -> None:
        super().__init__(config, FrameField.points)


class AnglePredictor(FeaturePredictor):
    def __init__(self, config: PredictorConfig) -> None:
        super().__init__(config, FrameField.angles)


class AngleVelPredictor(FeaturePredictor):
    def __init__(self, config: PredictorConfig) -> None:
        super().__init__(config, FrameField.angle_vel)


class AngleSymPredictor(FeaturePredictor):
    def __init__(self, config: PredictorConfig) -> None:
        super().__init__(config, FrameField.angle_sym)