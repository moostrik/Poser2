import numpy as np
import warnings
from dataclasses import replace

from modules.pose.Pose import Pose
from modules.pose.filter.PoseFilterBase import PoseFilterBase, PoseFilterConfigBase
from modules.pose.features import PoseFeatureData


class PoseValidatorConfig(PoseFilterConfigBase):
    """Configuration for PoseNanValidator."""

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


class PoseNanValidator(PoseFilterBase):
    """Validates that scores are 0 when values are NaN.

    Checks each feature type to ensure consistency: if a value is NaN, its score must be 0.
    Shows warnings when inconsistencies are found.
    Optionally fixes inconsistencies by setting scores to 0 when values are NaN.
    """

    def __init__(self, config: PoseValidatorConfig | None = None) -> None:
        self._config: PoseValidatorConfig = config or PoseValidatorConfig()

    def process(self, pose: Pose) -> Pose:
        """Validate all enabled features and show warnings if inconsistencies are found."""

        if self._config.validate_points:
            point_data = self._validate_feature(pose.point_data, "points")
            if point_data is not pose.point_data:
                pose = replace(pose, point_data=point_data)

        if self._config.validate_angles:
            angle_data = self._validate_feature(pose.angle_data, "angles")
            if angle_data is not pose.angle_data:
                pose = replace(pose, angle_data=angle_data)

        if self._config.validate_delta:
            delta_data = self._validate_feature(pose.delta_data, "delta")
            if delta_data is not pose.delta_data:
                pose = replace(pose, delta_data=delta_data)

        if self._config.validate_symmetry:
            symmetry_data = self._validate_feature(pose.symmetry_data, "symmetry")
            if symmetry_data is not pose.symmetry_data:
                pose = replace(pose, symmetry_data=symmetry_data)

        return pose

    def _validate_feature(self, data: PoseFeatureData, feature_name: str) -> PoseFeatureData:
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
            warnings.warn(f"Feature '{feature_name}' has {num_inconsistent}/{total_values} values with NaN but non-zero scores", UserWarning)

            # Fix inconsistencies if enabled
            if self._config.fix:
                fixed_scores: np.ndarray = scores.copy()
                fixed_scores[inconsistent] = 0.0
                # Return the same type as input with fixed scores
                return type(data)(values=values, scores=fixed_scores)

        return data