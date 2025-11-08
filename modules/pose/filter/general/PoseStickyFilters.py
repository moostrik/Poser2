"""Pose hold filters that replace NaN values with last valid values.

Provides "hold" behavior for angles, points, and deltas where NaN values
are replaced by the last valid value seen. Useful for maintaining continuity
when pose detection temporarily fails.
"""

# Standard library imports
from abc import abstractmethod
from dataclasses import replace

import numpy as np

# Pose imports
from modules.pose.filter.PoseFilterBase import PoseFilterBase, PoseFilterConfigBase
from modules.pose.Pose import Pose
from modules.pose.features import PoseFeatureData, PoseAngleData, PosePointData


class PoseStickyFilterConfig(PoseFilterConfigBase):
    """Configuration for pose hold filter with automatic change notification."""

    def __init__(self, init_to_zero: bool = False, hold_scores: bool = False) -> None:
        """
        Args:
            init_to_zero: If True, initialize with zeros to prevent NaN at start.
                         If False, first NaN values will remain NaN until valid data arrives.
            hold_scores: If True, preserve last valid scores when holding values.
                        If False, set scores to 0.0 for held (NaN) values.
        """
        super().__init__()
        self.init_to_zero: bool = init_to_zero
        self.hold_scores: bool = hold_scores


class PoseStickyFilterBase(PoseFilterBase):
    """Base class for pose hold filters.

    Replaces NaN values with the last valid value for each feature element.
    Maintains separate history for each joint/point as PoseFeatureData.

    Subclasses only need to specify:
    - Which feature to extract from pose (_get_feature_data)
    - How to replace feature data in pose (_replace_feature_data)
    - How to create empty feature data (_create_empty_feature_data)
    """

    def __init__(self, config: PoseStickyFilterConfig) -> None:
        self._config: PoseStickyFilterConfig = config
        self._last_valid: PoseFeatureData = self._initialize_last_valid()

    @property
    def config(self) -> PoseStickyFilterConfig:
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

    @abstractmethod
    def _create_empty_feature_data(self) -> PoseFeatureData:
        """Create empty feature data for initialization."""
        pass

    def _initialize_last_valid(self) -> PoseFeatureData:
        """Initialize last_valid based on config."""

        if self._config.init_to_zero:
            empty_data = self._create_empty_feature_data()
            values = np.zeros_like(empty_data.values)
            scores = np.ones_like(empty_data.scores)  # Valid scores for valid zeros
            return type(empty_data)(values=values, scores=scores)
        else:
            return self._create_empty_feature_data()

    def process(self, pose: Pose) -> Pose:
        """Replace NaN values with last valid values."""

        # Get feature data
        feature_data = self._get_feature_data(pose)

        # Use feature data's valid_mask (per-joint level)
        # valid_mask is True for valid joints (score > 0), False for invalid joints
        valid_mask = feature_data.valid_mask
        invalid_mask = ~valid_mask

        # Replace invalid values with last valid values
        # Need to broadcast invalid_mask to match values shape for points (17,) -> (17, 2)
        if feature_data.values.ndim > 1:
            # For 2D data (points), expand mask to match shape
            invalid_values_mask = invalid_mask[:, np.newaxis]
        else:
            # For 1D data (angles, deltas), use directly
            invalid_values_mask = invalid_mask

        held_values = np.where(invalid_values_mask, self._last_valid.values, feature_data.values)

        # Handle scores (always at joint level)
        if self._config.hold_scores:
            held_scores = np.where(invalid_mask, self._last_valid.scores, feature_data.scores)
        else:
            held_scores = np.where(invalid_mask, 0.0, feature_data.scores)

        held_scores = held_scores.astype(np.float32)

        # Update last valid (only where current is valid)
        if feature_data.values.ndim > 1:
            valid_values_mask = valid_mask[:, np.newaxis]
        else:
            valid_values_mask = valid_mask

        updated_values = np.where(valid_values_mask, feature_data.values, self._last_valid.values)
        updated_scores = np.where(valid_mask, feature_data.scores, self._last_valid.scores).astype(np.float32)

        # Store updated last valid data
        self._last_valid = type(feature_data)(values=updated_values, scores=updated_scores)

        # Create new feature data with held values
        held_data = type(feature_data)(values=held_values, scores=held_scores)

        # Return new pose with held feature
        return self._replace_feature_data(pose, held_data)

    def reset(self) -> None:
        """Reset the filter's internal state."""
        self._last_valid = self._initialize_last_valid()


class PoseAngleStickyFilter(PoseStickyFilterBase):
    """Holds last valid angle values when angles become NaN.

    Maintains continuity of joint angles when pose detection temporarily fails.
    """

    def _get_feature_data(self, pose: Pose) -> PoseFeatureData:
        return pose.angle_data

    def _replace_feature_data(self, pose: Pose, new_data: PoseFeatureData) -> Pose:
        return replace(pose, angle_data=new_data)

    def _create_empty_feature_data(self) -> PoseFeatureData:
        return PoseAngleData.create_empty()


class PosePointStickyFilter(PoseStickyFilterBase):
    """Holds last valid point coordinates when points become NaN.

    Maintains continuity of keypoint positions when pose detection temporarily fails.
    Handles 2D coordinates (x, y) per joint independently.
    """

    def _get_feature_data(self, pose: Pose) -> PoseFeatureData:
        return pose.point_data

    def _replace_feature_data(self, pose: Pose, new_data: PoseFeatureData) -> Pose:
        return replace(pose, point_data=new_data)

    def _create_empty_feature_data(self) -> PoseFeatureData:
        return PosePointData.create_empty()


class PoseDeltaStickyFilter(PoseStickyFilterBase):
    """Holds last valid delta values when deltas become NaN.

    Maintains continuity of angle changes when pose detection temporarily fails.
    """

    def _get_feature_data(self, pose: Pose) -> PoseFeatureData:
        return pose.delta_data

    def _replace_feature_data(self, pose: Pose, new_data: PoseFeatureData) -> Pose:
        return replace(pose, delta_data=new_data)

    def _create_empty_feature_data(self) -> PoseFeatureData:
        return PoseAngleData.create_empty()


class PoseStickyFilter(PoseFilterBase):
    """Holds last valid values for all pose features (angles, points, and deltas).

    Applies the same hold configuration to all features. For independent
    control of each feature, use PoseAngleHoldFilter, PosePointHoldFilter, and
    PoseDeltaHoldFilter separately.
    """

    def __init__(self, config: PoseStickyFilterConfig) -> None:
        self._config: PoseStickyFilterConfig = config

        # Create individual hold filters for each feature
        self._angle_filter = PoseAngleStickyFilter(config)
        self._point_filter = PosePointStickyFilter(config)
        self._delta_filter = PoseDeltaStickyFilter(config)

    @property
    def config(self) -> PoseStickyFilterConfig:
        """Access the filter's configuration."""
        return self._config

    def process(self, pose: Pose) -> Pose:
        """Hold last valid values for all features in the pose."""
        pose = self._angle_filter.process(pose)
        pose = self._point_filter.process(pose)
        pose = self._delta_filter.process(pose)
        return pose

    def reset(self) -> None:
        """Reset all hold filters' internal state."""
        self._angle_filter.reset()
        self._point_filter.reset()
        self._delta_filter.reset()


__all__: list[str] = [
    'PoseStickyFilterConfig',
    'PoseStickyFilterBase',
    'PoseAngleStickyFilter',
    'PosePointStickyFilter',
    'PoseDeltaStickyFilter',
    'PoseStickyFilter',
]