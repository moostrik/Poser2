"""Pose confidence filters for removing low-confidence values.

Provides confidence-based filtering for angles, points, and deltas where values
below threshold are set to NaN and scores are rescaled to [0, 1] range.
"""

# Standard library imports
from dataclasses import replace

import numpy as np

# Pose imports
from modules.pose.nodes.Nodes import FilterNode, NodeConfigBase
from modules.pose.Frame import Frame, FrameField


class ConfidenceFilterConfig(NodeConfigBase):
    """Configuration for confidence filtering with automatic change notification."""

    def __init__(self, confidence_threshold: float = 0.5, rescale_scores: bool = True) -> None:
        """
        Args:
            confidence_threshold: Minimum confidence score to keep values.
                                Values below this are set to NaN.
                                Range: [0.0, 0.99] (clamped automatically)
            rescale_scores: If True, rescale remaining scores to [0, 1] range.
                          If False, keep original scores for values above threshold.
        """
        super().__init__()
        self.confidence_threshold: float = max(0.0, min(0.99, confidence_threshold))
        self.rescale_scores: bool = rescale_scores


class ConfidenceFilter(FilterNode):
    """Generic confidence filter for pose features.

    Filters low-confidence values by:
    1. Setting values to NaN when score < threshold
    2. Optionally rescaling remaining scores to [0, 1] range

    Args:
        config: Confidence filter configuration
        feature_class: Feature class type (e.g., AngleFeature, Point2DFeature)
        attr_name: Name of the pose attribute to filter

    Example:
        filter = FeatureConfidenceFilter(config, AngleFeature, "angles")
        filter = FeatureConfidenceFilter(config, Point2DFeature, "points")
    """

    def __init__(self, config: ConfidenceFilterConfig, pose_field: FrameField):
        self._config = config
        self._pose_field = pose_field

    @property
    def config(self) -> ConfidenceFilterConfig:
        return self._config

    def process(self, pose: Frame) -> Frame:
        """Filter values based on confidence threshold."""
        feature_data = pose.get_feature(self._pose_field)

        # Skip if no valid values
        if feature_data.valid_count == 0:
            return pose

        values: np.ndarray = feature_data.values
        scores: np.ndarray = feature_data.scores

        # Mask for values meeting confidence threshold
        filtered = scores >= self._config.confidence_threshold

        # Replace low-confidence values with NaN
        # For 2D data (points), broadcast mask to match shape (17,) -> (17, 2)
        if values.ndim > 1:
            filtered_mask = filtered[:, np.newaxis]
        else:
            filtered_mask = filtered

        filtered_values: np.ndarray = np.where(filtered_mask, values, np.nan)

        # Handle scores based on rescale_scores config
        if self._config.rescale_scores:
            # Rescale remaining scores to [0, 1] range
            # Division by (1 - threshold) remaps [threshold, 1.0] -> [0, 1.0]
            filtered_scores: np.ndarray = np.where(
                filtered,
                (scores - self._config.confidence_threshold) / (1.0 - self._config.confidence_threshold),
                0.0
            ).astype(np.float32)
        else:
            # Keep original scores for values above threshold, set to 0 for filtered values
            filtered_scores: np.ndarray = np.where(filtered, scores, 0.0).astype(np.float32)

        # Create new feature data with filtered values
        filtered_data = type(feature_data)(values=filtered_values, scores=filtered_scores)

        # Return new pose with filtered feature
        return replace(pose, **{self._pose_field.name: filtered_data})


# Convenience classes
class BBoxConfidenceFilter(ConfidenceFilter):
    def __init__(self, config: ConfidenceFilterConfig) -> None:
        super().__init__(config, FrameField.bbox)


class PointConfidenceFilter(ConfidenceFilter):
    def __init__(self, config: ConfidenceFilterConfig) -> None:
        super().__init__(config, FrameField.points)


class AngleConfidenceFilter(ConfidenceFilter):
    def __init__(self, config: ConfidenceFilterConfig) -> None:
        super().__init__(config, FrameField.angles)


class AngleVelConfidenceFilter(ConfidenceFilter):
    def __init__(self, config: ConfidenceFilterConfig) -> None:
        super().__init__(config, FrameField.angle_vel)


class AngleSymConfFilter(ConfidenceFilter):
    def __init__(self, config: ConfidenceFilterConfig) -> None:
        super().__init__(config, FrameField.angle_sym)
