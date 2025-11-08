"""Pose validators for data integrity checks."""

import warnings
from abc import abstractmethod
from dataclasses import replace

import numpy as np

from modules.pose.Pose import Pose
from modules.pose.filters.FilterBase import FilterBase, FilterConfigBase
from modules.pose.features import POSE_FEATURE_RANGES, POSE_CLASS_TO_FEATURE_TYPE, PoseFeatureData


class ValidatorConfig(FilterConfigBase):
    """Configuration for pose validators."""

    def __init__(
        self,
        validate_points: bool = True,
        validate_angles: bool = True,
        validate_delta: bool = True,
        validate_symmetry: bool = True,
        fix: bool = True
    ) -> None:
        super().__init__()
        self.validate_points: bool = validate_points
        self.validate_angles: bool = validate_angles
        self.validate_delta: bool = validate_delta
        self.validate_symmetry: bool = validate_symmetry
        self.fix: bool = fix


class ValidatorBase(FilterBase):
    """Base class for pose validators.

    Provides common validation pattern for all feature types.
    Subclasses implement specific validation logic via _validate_feature_data().
    """

    def __init__(self, config: ValidatorConfig | None = None) -> None:
        self._config: ValidatorConfig = config or ValidatorConfig()

    @property
    def config(self) -> ValidatorConfig:
        """Access the validator's configuration."""
        return self._config

    def process(self, pose: Pose) -> Pose:
        """Validate all enabled features."""

        if self._config.validate_points:
            point_data = self._validate_feature_data(pose.point_data, "points")
            if point_data is not pose.point_data:
                pose = replace(pose, point_data=point_data)

        if self._config.validate_angles:
            angle_data = self._validate_feature_data(pose.angle_data, "angles")
            if angle_data is not pose.angle_data:
                pose = replace(pose, angle_data=angle_data)

        if self._config.validate_delta:
            delta_data = self._validate_feature_data(pose.delta_data, "delta")
            if delta_data is not pose.delta_data:
                pose = replace(pose, delta_data=delta_data)

        if self._config.validate_symmetry:
            symmetry_data = self._validate_feature_data(pose.symmetry_data, "symmetry")
            if symmetry_data is not pose.symmetry_data:
                pose = replace(pose, symmetry_data=symmetry_data)

        return pose

    @abstractmethod
    def _validate_feature_data(self, data: PoseFeatureData, feature_name: str) -> PoseFeatureData:
        """Validate a single feature's data.

        Args:
            data: Feature data to validate
            feature_name: Name for warning messages

        Returns:
            Original or fixed feature data
        """
        pass


class NanValidator(ValidatorBase):
    """Validates that scores are 0 when values are NaN.

    Checks each feature type to ensure consistency: if a value is NaN, its score must be 0.
    Shows warnings when inconsistencies are found.
    Optionally fixes inconsistencies by setting scores to 0 when values are NaN.
    """

    def _validate_feature_data(self, data: PoseFeatureData, feature_name: str) -> PoseFeatureData:
        """Validate that scores are 0 when values are NaN."""
        values: np.ndarray = data.values

        # Only validate if feature has scores attribute
        if not hasattr(data, 'scores'):
            return data

        scores: np.ndarray = data.scores

        # Find NaN values - for 2D data (points), check if ANY dimension is NaN per row
        has_nan: np.ndarray = np.any(np.isnan(values), axis=-1) if values.ndim > 1 else np.isnan(values)

        # Check if scores are non-zero when values are NaN
        inconsistent = has_nan & (scores > 0)

        if np.any(inconsistent):
            num_inconsistent = int(np.sum(inconsistent))
            total_values: int = len(scores) if scores.ndim == 1 else scores.size
            warnings.warn(
                f"Feature '{feature_name}' has {num_inconsistent}/{total_values} values with NaN but non-zero scores",
                UserWarning
            )

            # Fix inconsistencies if enabled
            if self._config.fix:
                fixed_scores: np.ndarray = scores.copy()
                fixed_scores[inconsistent] = 0.0
                return type(data)(values=values, scores=fixed_scores)

        return data


class RangeValidator(ValidatorBase):
    """Validates that pose feature values are within expected ranges.

    Checks each feature type against its defined range from POSE_FEATURE_RANGES.
    Shows warnings when values are out of range.
    Optionally fixes out-of-range values by clamping them to the valid range.
    """

    def _validate_feature_data(self, data: PoseFeatureData, feature_name: str) -> PoseFeatureData:
        """Validate feature values are within expected range."""
        # Get feature type from data's class type using reverse lookup
        data_type = type(data)
        feature_type = POSE_CLASS_TO_FEATURE_TYPE.get(data_type)

        if feature_type is None:
            return data

        min_val, max_val = POSE_FEATURE_RANGES[feature_type]
        values: np.ndarray = data.values

        # Check if any finite values are out of range (ignore NaN)
        # For 2D data (points), check if ANY coordinate is out of range per joint
        out_of_range_elements: np.ndarray = np.isfinite(values) & ((values < min_val) | (values > max_val))

        # For multi-dimensional data, check if any element in each row is out of range
        if values.ndim > 1:
            out_of_range: np.ndarray = np.any(out_of_range_elements, axis=-1)  # Per joint
            num_invalid = int(np.sum(out_of_range))  # Count joints
            total_count = values.shape[0]  # Number of joints
        else:
            out_of_range = out_of_range_elements
            num_invalid = int(np.sum(out_of_range))  # Count values
            total_count = values.size  # Number of values

        if np.any(out_of_range):
            warnings.warn(
                f"Feature '{feature_name}' has {num_invalid}/{total_count} "
                f"{'joints' if values.ndim > 1 else 'values'} out of range [{min_val}, {max_val}]",
                UserWarning
            )

            # Fix out-of-range values if enabled
            if self._config.fix:
                fixed_values: np.ndarray = values.copy()
                # Clamp all finite values (element-wise, works for both 1D and 2D)
                finite_mask: np.ndarray = np.isfinite(fixed_values)
                fixed_values[finite_mask] = np.clip(fixed_values[finite_mask], min_val, max_val)
                return type(data)(values=fixed_values, scores=data.scores)

        return data


class ScoreValidator(ValidatorBase):
    """Validates that confidence scores are in [0.0, 1.0]."""

    def _validate_feature_data(self, data: PoseFeatureData, feature_name: str) -> PoseFeatureData:
        if not hasattr(data, 'scores'):
            return data

        invalid = (data.scores < 0.0) | (data.scores > 1.0)
        if np.any(invalid):
            num_invalid = int(np.sum(invalid))
            warnings.warn(
                f"Feature '{feature_name}' has {num_invalid}/{len(data.scores)} scores outside [0.0, 1.0]",
                UserWarning
            )
            if self._config.fix:
                fixed_scores = np.clip(data.scores, 0.0, 1.0)
                return type(data)(values=data.values, scores=fixed_scores)
        return data


class PoseValidator(FilterBase):
    """Validates all pose features (points, angles, delta, symmetry) for data integrity.

    Combines NaN, range, and score validation. For independent control of each
    validation type, use PoseNanValidator, PoseRangeValidator, and PoseScoreValidator separately.
    """

    def __init__(self, config: ValidatorConfig | None = None) -> None:
        self._config: ValidatorConfig = config or ValidatorConfig()

        # Create individual validators
        self._nan_validator = NanValidator(self._config)
        self._range_validator = RangeValidator(self._config)
        self._score_validator = ScoreValidator(self._config)

    @property
    def config(self) -> ValidatorConfig:
        """Access the validator's configuration."""
        return self._config

    def process(self, pose: Pose) -> Pose:
        """Validate all features with all validation types."""
        pose = self._nan_validator.process(pose)
        pose = self._range_validator.process(pose)
        pose = self._score_validator.process(pose)
        return pose