"""Pose confidence filters for removing low-confidence values.

Provides confidence-based filtering for angles, points, and deltas where values
below threshold are set to NaN and scores are rescaled to [0, 1] range.
"""

# Standard library imports
import numpy as np

# Pose imports
from ...features import Angles, AngleVelocity, AngleSymmetry, BBox, Points2D, BaseFeature
from ..nodes import FilterNode
from ...frame import Frame, replace
from modules.settings import BaseSettings, Field


class ConfidenceFilterSettings(BaseSettings):
    """Configuration for confidence filtering."""
    confidence_threshold: Field[float] = Field(0.5)
    rescale_scores:       Field[bool]  = Field(True)


class ConfidenceFilter(FilterNode):
    """Generic confidence filter for pose features.

    Filters low-confidence values by:
    1. Setting values to NaN when score < threshold
    2. Optionally rescaling remaining scores to [0, 1] range

    Args:
        settings: Confidence filter configuration
        feature_class: Feature class type (e.g., AngleFeature, Point2DFeature)
        attr_name: Name of the pose attribute to filter

    Example:
        filter = FeatureConfidenceFilter(settings, AngleFeature, "angles")
        filter = FeatureConfidenceFilter(settings, Point2DFeature, "points")
    """

    def __init__(self, settings: ConfidenceFilterSettings, feature_type: type[BaseFeature]):
        self.settings = settings
        self._feature_type = feature_type

    def process(self, pose: Frame) -> Frame:
        """Filter values based on confidence threshold."""
        feature_data = pose[self._feature_type]

        # Skip if no valid values
        if feature_data.valid_count == 0:
            return pose

        values: np.ndarray = feature_data.values
        scores: np.ndarray = feature_data.scores

        # Mask for values meeting confidence threshold
        filtered = scores >= self.settings.confidence_threshold

        # Replace low-confidence values with NaN
        # For 2D data (points), broadcast mask to match shape (17,) -> (17, 2)
        if values.ndim > 1:
            filtered_mask = filtered[:, np.newaxis]
        else:
            filtered_mask = filtered

        filtered_values: np.ndarray = np.where(filtered_mask, values, np.nan)

        # Handle scores based on rescale_scores setting
        if self.settings.rescale_scores:
            # Rescale remaining scores to [0, 1] range
            # Division by (1 - threshold) remaps [threshold, 1.0] -> [0, 1.0]
            filtered_scores: np.ndarray = np.where(
                filtered,
                (scores - self.settings.confidence_threshold) / (1.0 - self.settings.confidence_threshold),
                0.0
            ).astype(np.float32)
        else:
            # Keep original scores for values above threshold, set to 0 for filtered values
            filtered_scores: np.ndarray = np.where(filtered, scores, 0.0).astype(np.float32)

        # Create new feature data with filtered values
        filtered_data = type(feature_data)(values=filtered_values, scores=filtered_scores)

        # Return new pose with filtered feature
        return replace(pose, {self._feature_type: filtered_data})


# Convenience classes
class BBoxConfidenceFilter(ConfidenceFilter):
    def __init__(self, settings:ConfidenceFilterSettings) -> None:
        super().__init__(settings,BBox)


class PointConfidenceFilter(ConfidenceFilter):
    def __init__(self, settings:ConfidenceFilterSettings) -> None:
        super().__init__(settings,Points2D)


class AngleConfidenceFilter(ConfidenceFilter):
    def __init__(self, settings:ConfidenceFilterSettings) -> None:
        super().__init__(settings,Angles)


class AngleVelocityConfidenceFilter(ConfidenceFilter):
    def __init__(self, settings:ConfidenceFilterSettings) -> None:
        super().__init__(settings,AngleVelocity)


class AngleSymmetryConfidenceFilter(ConfidenceFilter):
    def __init__(self, settings:ConfidenceFilterSettings) -> None:
        super().__init__(settings,AngleSymmetry)
