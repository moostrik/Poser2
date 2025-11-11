"""Pose confidence filters for removing low-confidence values.

Provides confidence-based filtering for angles, points, and deltas where values
below threshold are set to NaN and scores are rescaled to [0, 1] range.
"""

# Standard library imports
from abc import abstractmethod
from dataclasses import replace

import numpy as np

# Pose imports
from modules.pose.Nodes import FilterNode, NodeConfigBase
from modules.pose.Pose import Pose
from modules.pose.features import PoseFeatureData


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


class ConfidenceFilterBase(FilterNode):
    """Base class for pose confidence filters.

    Filters low-confidence values by:
    1. Setting values to NaN when score < threshold
    2. Optionally rescaling remaining scores to [0, 1] range

    Subclasses only need to specify:
    - Which feature to extract from pose (_get_feature_data)
    - How to replace feature data in pose (_replace_feature_data)
    """

    def __init__(self, config: ConfidenceFilterConfig) -> None:
        self._config: ConfidenceFilterConfig = config

    @property
    def config(self) -> ConfidenceFilterConfig:
        """Access the filter's configuration."""
        return self._config

    @abstractmethod
    def _get_feature_data(self, pose: Pose) -> PoseFeatureData:
        """Extract the feature data to process from the pose."""
        pass

    @abstractmethod
    def _replace_feature_data(self, pose: Pose, new_data: PoseFeatureData) -> Pose:
        """Create new pose with replaced feature data."""
        pass

    def process(self, pose: Pose) -> Pose:
        """Filter values based on confidence threshold."""

        # Get feature data
        feature_data = self._get_feature_data(pose)

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
        return self._replace_feature_data(pose, filtered_data)


class AngleConfidenceFilter(ConfidenceFilterBase):
    """Filters angle values based on confidence scores.

    Sets low-confidence angles to NaN and rescales remaining scores.
    """

    def _get_feature_data(self, pose: Pose) -> PoseFeatureData:
        return pose.angles

    def _replace_feature_data(self, pose: Pose, new_data: PoseFeatureData) -> Pose:
        return replace(pose, angles=new_data)


class PointConfidenceFilter(ConfidenceFilterBase):
    """Filters point coordinates based on confidence scores.

    Sets low-confidence points to NaN and rescales remaining scores.
    Handles 2D coordinates (x, y) per joint.
    """

    def _get_feature_data(self, pose: Pose) -> PoseFeatureData:
        return pose.points

    def _replace_feature_data(self, pose: Pose, new_data: PoseFeatureData) -> Pose:
        return replace(pose, points=new_data)


class DeltaConfidenceFilter(ConfidenceFilterBase):
    """Filters delta values based on confidence scores.

    Sets low-confidence deltas to NaN and rescales remaining scores.
    """

    def _get_feature_data(self, pose: Pose) -> PoseFeatureData:
        return pose.deltas

    def _replace_feature_data(self, pose: Pose, new_data: PoseFeatureData) -> Pose:
        return replace(pose, deltas=new_data)


class PoseConfidenceFilter(FilterNode):
    """Filters all pose features (angles, points, and deltas) based on confidence.

    Applies the same confidence threshold to all features. For independent
    control of each feature, use PoseAngleConfidenceFilter, PosePointConfidenceFilter,
    and PoseDeltaConfidenceFilter separately.
    """

    def __init__(self, config: ConfidenceFilterConfig) -> None:
        self._config: ConfidenceFilterConfig = config

        # Create individual confidence filters for each feature
        self._angle_filter = AngleConfidenceFilter(config)
        self._point_filter = PointConfidenceFilter(config)
        self._delta_filter = DeltaConfidenceFilter(config)

    @property
    def config(self) -> ConfidenceFilterConfig:
        """Access the filter's configuration."""
        return self._config

    def process(self, pose: Pose) -> Pose:
        """Filter all features based on confidence threshold."""
        pose = self._angle_filter.process(pose)
        pose = self._point_filter.process(pose)
        pose = self._delta_filter.process(pose)
        return pose