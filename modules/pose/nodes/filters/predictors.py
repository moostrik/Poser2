"""Pose prediction filters: a feature extrapolated ``lead`` seconds past its frame's stamp.

The fit (linear or quadratic) is on the last ``fit_frames`` frames and their stamps, so a dropped camera frame
leaves the velocity as it is; angles wrap and points clamp.
"""

# Standard library imports
from collections import defaultdict

# Third-party imports
import numpy as np

# Pose imports
from ...features import Angles, Azimuth, BBox, Points2D, AngleVelocity, AngleSymmetry, BaseFeature
from .._utils.array_predict import AnglePredict, PointPredict, Predict, PredictionMethod
from ..nodes import FilterNode
from ...frame import Frame, replace
from modules.settings import BaseSettings, Field


class PredictorSettings(BaseSettings):
    """Configuration for pose prediction."""
    method:     Field[PredictionMethod] = Field(PredictionMethod.QUADRATIC)
    fit_frames: Field[int]   = Field(3, min=2, max=10, description="Frames the fit spans")
    lead:       Field[float] = Field(1.0 / 30.0, min=0.0, max=0.5, step=0.001, description="Seconds past the frame's stamp the pose is extrapolated to")


class FeaturePredictor(FilterNode):
    """Generic pose feature predictor."""

    _PREDICT_MAP: dict[type[BaseFeature], type] = defaultdict(
        lambda: Predict,
        {
            Angles: AnglePredict,
            Azimuth: AnglePredict,
            Points2D: PointPredict,
        }
    )

    def __init__(self, settings: PredictorSettings, feature_type: type[BaseFeature]) -> None:
        self.settings = settings
        self._feature_type = feature_type
        predictor_cls = self._PREDICT_MAP[feature_type]
        self._predictor = predictor_cls(
            vector_size=feature_type.length(),
            method=settings.method,
            fit_frames=settings.fit_frames,
            clamp_range=feature_type.range()
        )
        self.settings.bind_all(self._on_settings_changed)

    def __del__(self):
        """Cleanup settings listener to prevent memory leaks."""
        try:
            self.settings.unbind_all(self._on_settings_changed)
        except (AttributeError, ValueError):
            pass

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
        """Add the frame's feature to the fit and return the pose with the feature ``lead`` seconds ahead."""
        feature_data = pose[self._feature_type]

        self._predictor.add_sample(feature_data.values, pose.timestamp)
        predicted_values: np.ndarray = self._predictor.at(pose.timestamp + self.settings.lead)

        # Create new feature data with predicted values
        predicted_data = self._create_predicted_data(feature_data, predicted_values)

        # Return new pose with predicted feature
        return replace(pose, {self._feature_type: predicted_data})

    def reset(self) -> None:
        """Reset the predictor's internal state (clear sample history)."""
        self._predictor.reset()

    def _on_settings_changed(self, _=None) -> None:
        """Handle configuration changes by updating predictor parameters."""
        self._predictor.method = self.settings.method
        self._predictor.fit_frames = self.settings.fit_frames


# Convenience classes

class BBoxPredictor(FeaturePredictor):
    def __init__(self, settings:PredictorSettings) -> None:
        super().__init__(settings,BBox)


class PointPredictor(FeaturePredictor):
    def __init__(self, settings:PredictorSettings) -> None:
        super().__init__(settings,Points2D)


class AnglePredictor(FeaturePredictor):
    def __init__(self, settings:PredictorSettings) -> None:
        super().__init__(settings,Angles)


class AngleVelocityPredictor(FeaturePredictor):
    def __init__(self, settings:PredictorSettings) -> None:
        super().__init__(settings,AngleVelocity)


class AngleSymmetryPredictor(FeaturePredictor):
    def __init__(self, settings:PredictorSettings) -> None:
        super().__init__(settings,AngleSymmetry)


class AzimuthPredictor(FeaturePredictor):
    def __init__(self, settings:PredictorSettings) -> None:
        super().__init__(settings,Azimuth)
