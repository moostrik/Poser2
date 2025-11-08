"""Pose prediction filters for extrapolating future pose states.

Provides prediction for angles, points, and deltas using linear or quadratic
extrapolation with proper handling of circular values and coordinate clamping.
"""

# Standard library imports
from abc import abstractmethod
from dataclasses import replace

import numpy as np

# Pose imports
from modules.pose.filter.PoseFilterBase import PoseFilterBase, PoseFilterConfigBase
from modules.pose.Pose import Pose
from modules.pose.filter.general.VectorMath.VectorPredictors import Predictor, AnglePredictor, PointPredictor, PredictionMethod
from modules.pose.features import PoseFeatureData, ANGLE_NUM_JOINTS, POSE_NUM_JOINTS, POSE_POINTS_RANGE


class PosePredictorConfig(PoseFilterConfigBase):
    """Configuration for pose prediction with automatic change notification."""

    def __init__(self, frequency: float = 30.0, method: PredictionMethod = PredictionMethod.QUADRATIC) -> None:
        super().__init__()
        self.frequency: float = frequency
        self.method: PredictionMethod = method


class PosePredictorBase(PoseFilterBase):
    """Base class for pose predictors.

    Handles common prediction logic. Subclasses only need to specify:
    - Which predictor instance to create
    - Which feature to extract/replace from pose
    - How to reconstruct feature data with predictions

    Note: Predictions preserve original confidence scores for valid values,
    but set scores to 0 where predictions are NaN (insufficient history).
    When predictor has insufficient samples, all predictions will be NaN
    and all scores will be 0.
    """

    def __init__(self, config: PosePredictorConfig) -> None:
        self._config: PosePredictorConfig = config
        self._predictor: Predictor
        self._initialize_predictor()
        self._config.add_listener(self._on_config_changed)

    @property
    def config(self) -> PosePredictorConfig:
        """Access the predictor's configuration."""
        return self._config

    @abstractmethod
    def _initialize_predictor(self) -> None:
        """Create the appropriate predictor instance."""
        pass

    @abstractmethod
    def _get_feature_data(self, pose: Pose) -> PoseFeatureData:
        """Extract the feature data to predict from the pose."""
        pass

    @abstractmethod
    def _create_predicted_data(self, original_data: PoseFeatureData, predicted_values: np.ndarray) -> PoseFeatureData:
        """Create new feature data with predicted values."""
        pass

    @abstractmethod
    def _replace_feature_data(self, pose: Pose, new_data: PoseFeatureData) -> Pose:
        """Create new pose with replaced feature data."""
        pass

    def process(self, pose: Pose) -> Pose:
        """Add current feature data to predictor and return pose with predicted values."""

        # Get feature data
        feature_data = self._get_feature_data(pose)

        # Add sample and get prediction
        self._predictor.add_sample(feature_data.values)
        predicted_values: np.ndarray = self._predictor.value

        # Create new feature data with predicted values
        predicted_data = self._create_predicted_data(feature_data, predicted_values)

        # Return new pose with predicted feature
        return self._replace_feature_data(pose, predicted_data)

    def reset(self) -> None:
        """Reset the predictor's internal state (clear sample history)."""
        self._predictor.reset()

    def _on_config_changed(self) -> None:
        """Handle configuration changes by updating predictor parameters."""
        self._predictor.input_frequency = self._config.frequency
        self._predictor.method = self._config.method


class PoseAnglePredictor(PosePredictorBase):
    """Predicts angle data for the next frame using vectorized angle prediction.

    Uses AnglePredictor which handles circular wrapping of angle values.
    """

    def _initialize_predictor(self) -> None:
        self._predictor = AnglePredictor(vector_size=ANGLE_NUM_JOINTS, input_frequency=self._config.frequency, method=self._config.method)

    def _get_feature_data(self, pose: Pose) -> PoseFeatureData:
        return pose.angle_data

    def _create_predicted_data(self, original_data: PoseFeatureData, predicted_values: np.ndarray) -> PoseFeatureData:
        """Create angle data with predicted values and adjusted scores.

        Sets scores to 0 where predictions are NaN, preserves original scores otherwise.
        """
        has_nan: np.ndarray = np.isnan(predicted_values)
        interpolated_scores: np.ndarray = np.where(has_nan, 0.0, original_data.scores).astype(np.float32)
        return type(original_data)(values=predicted_values, scores=interpolated_scores)

    def _replace_feature_data(self, pose: Pose, new_data: PoseFeatureData) -> Pose:
        return replace(pose, angle_data=new_data)


class PosePointPredictor(PosePredictorBase):
    """Predicts point data for the next frame using vectorized point prediction.

    Uses PointPredictor which clamps coordinates to [0, 1] range and handles 2D data.
    """

    def _initialize_predictor(self) -> None:
        self._predictor = PointPredictor(num_points=POSE_NUM_JOINTS, input_frequency=self._config.frequency, method=self._config.method, clamp_range=POSE_POINTS_RANGE)

    def _get_feature_data(self, pose: Pose) -> PoseFeatureData:
        return pose.point_data

    def _create_predicted_data(self, original_data: PoseFeatureData, predicted_values: np.ndarray) -> PoseFeatureData:
        """Create point data with predicted values and adjusted scores.

        Checks if ANY coordinate (x or y) is NaN per joint.
        Sets scores to 0 for joints with NaN predictions, preserves original scores otherwise.
        """
        has_nan: np.ndarray = np.any(np.isnan(predicted_values), axis=-1)
        interpolated_scores: np.ndarray = np.where(has_nan, 0.0, original_data.scores).astype(np.float32)
        return type(original_data)(values=predicted_values, scores=interpolated_scores)

    def _replace_feature_data(self, pose: Pose, new_data: PoseFeatureData) -> Pose:
        return replace(pose, point_data=new_data)


class PoseDeltaPredictor(PosePredictorBase):
    """Predicts delta data for the next frame using vectorized angle prediction.

    Uses AnglePredictor since delta represents angle changes (circular values).
    """

    def _initialize_predictor(self) -> None:
        self._predictor = AnglePredictor(vector_size=ANGLE_NUM_JOINTS, input_frequency=self._config.frequency, method=self._config.method)

    def _get_feature_data(self, pose: Pose) -> PoseFeatureData:
        return pose.delta_data

    def _create_predicted_data(self, original_data: PoseFeatureData, predicted_values: np.ndarray) -> PoseFeatureData:
        """Create delta data with predicted values and adjusted scores.

        Sets scores to 0 where predictions are NaN, preserves original scores otherwise.
        """
        has_nan: np.ndarray = np.isnan(predicted_values)
        interpolated_scores: np.ndarray = np.where(has_nan, 0.0, original_data.scores).astype(np.float32)
        return type(original_data)(values=predicted_values, scores=interpolated_scores)

    def _replace_feature_data(self, pose: Pose, new_data: PoseFeatureData) -> Pose:
        return replace(pose, delta_data=new_data)


class PosePredictor(PoseFilterBase):
    """Predicts all pose features (angles, points, and deltas) for the next frame.

    Applies the same prediction configuration to all features. For independent
    control of each feature, use PoseAnglePredictor, PosePointPredictor, and
    PoseDeltaPredictor separately.
    """

    def __init__(self, config: PosePredictorConfig) -> None:
        self._config: PosePredictorConfig = config

        # Create individual predictors for each feature
        self._angle_predictor = PoseAnglePredictor(config)
        self._point_predictor = PosePointPredictor(config)
        self._delta_predictor = PoseDeltaPredictor(config)

    @property
    def config(self) -> PosePredictorConfig:
        """Access the predictor's configuration."""
        return self._config

    def process(self, pose: Pose) -> Pose:
        """Predict all features in the pose."""
        pose = self._angle_predictor.process(pose)
        pose = self._point_predictor.process(pose)
        pose = self._delta_predictor.process(pose)
        return pose

    def reset(self) -> None:
        """Reset all predictors' internal state."""
        self._angle_predictor.reset()
        self._point_predictor.reset()
        self._delta_predictor.reset()