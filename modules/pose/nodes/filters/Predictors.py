"""Pose prediction filters for extrapolating future pose states.

Provides prediction for angles, points, and deltas using linear or quadratic
extrapolation with proper handling of circular values and coordinate clamping.
"""

# Standard library imports
from dataclasses import replace

# Third-party imports
import numpy as np

# Pose imports
from modules.pose.features import Angles, BBox, Points2D, Symmetry
from modules.pose.nodes._utils.VectorPredict import AnglePredict, PointPredict, VectorPredict, PredictionMethod
from modules.pose.nodes.Nodes import FilterNode, NodeConfigBase
from modules.pose.Pose import Pose


class PredictorConfig(NodeConfigBase):
    """Configuration for pose prediction with automatic change notification."""

    def __init__(self, frequency: float = 30.0, method: PredictionMethod = PredictionMethod.QUADRATIC) -> None:
        super().__init__()
        self.frequency: float = frequency
        self.method: PredictionMethod = method


class FeaturePredictor(FilterNode):
    """Generic pose feature predictor.

    Args:
        config: Predictor configuration
        feature_class: Feature class type (e.g., AngleFeature, Point2DFeature)
        attr_name: Name of the pose attribute to predict

    Example:
        predictor = FeaturePredictor(config, AngleFeature, "angles")
        predictor = FeaturePredictor(config, Point2DFeature, "points")
    """

    # Registry mapping feature classes to predictor classes
    PREDICTOR_REGISTRY = {
        Angles: AnglePredict,
        BBox: VectorPredict,
        Points2D: PointPredict,
        Symmetry: VectorPredict,
    }

    def __init__(self, config: PredictorConfig, feature_class: type, attr_name: str):
        if feature_class not in self.PREDICTOR_REGISTRY:
            valid_classes = [cls.__name__ for cls in self.PREDICTOR_REGISTRY.keys()]
            raise ValueError(
                f"Unknown feature class '{feature_class.__name__}'. "
                f"Must be one of: {valid_classes}"
            )

        self._config = config
        self._attr_name = attr_name
        self._feature_class = feature_class

        predictor_cls = self.PREDICTOR_REGISTRY[feature_class]
        self._predictor = predictor_cls(
            vector_size=len(feature_class.feature_enum()),
            input_frequency=config.frequency,
            method=config.method,
            clamp_range=feature_class.default_range()
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

    def process(self, pose: Pose) -> Pose:
        """Add current feature data to predictor and return pose with predicted values."""
        feature_data = getattr(pose, self._attr_name)

        # Add sample and get prediction
        self._predictor.add_sample(feature_data.values)
        predicted_values: np.ndarray = self._predictor.value

        # Create new feature data with predicted values
        predicted_data = self._create_predicted_data(feature_data, predicted_values)

        # Return new pose with predicted feature
        return replace(pose, **{self._attr_name: predicted_data})

    def reset(self) -> None:
        """Reset the predictor's internal state (clear sample history)."""
        self._predictor.reset()

    def _on_config_changed(self) -> None:
        """Handle configuration changes by updating predictor parameters."""
        self._predictor.input_frequency = self._config.frequency
        self._predictor.method = self._config.method


# Convenience classes
class AnglePredictor(FeaturePredictor):
    def __init__(self, config: PredictorConfig) -> None:
        super().__init__(config, Angles, "angles")


class BBoxPredictor(FeaturePredictor):
    def __init__(self, config: PredictorConfig) -> None:
        super().__init__(config, BBox, "bbox")


class DeltaPredictor(FeaturePredictor):
    def __init__(self, config: PredictorConfig) -> None:
        super().__init__(config, Angles, "deltas")


class PointPredictor(FeaturePredictor):
    def __init__(self, config: PredictorConfig) -> None:
        super().__init__(config, Points2D, "points")


class SymmetryPredictor(FeaturePredictor):
    def __init__(self, config: PredictorConfig) -> None:
        super().__init__(config, Symmetry, "symmetry")