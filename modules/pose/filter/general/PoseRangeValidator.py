import numpy as np
import warnings
from dataclasses import replace

from modules.pose.Pose import Pose
from modules.pose.filter.PoseFilterBase import PoseFilterBase, PoseFilterConfigBase
from modules.pose.features import PoseFeatureType, POSE_FEATURE_RANGES, PoseFeatureData


class PoseRangeValidatorConfig(PoseFilterConfigBase):
    """Configuration for PoseRangeValidator."""

    def __init__(
        self,
        validate_points: bool = True,
        validate_angles: bool = True,
        validate_delta: bool = True,
        validate_symmetry: bool = True,
        fix: bool = False
    ) -> None:
        super().__init__()
        self.validate_points: bool = validate_points
        self.validate_angles: bool = validate_angles
        self.validate_delta: bool = validate_delta
        self.validate_symmetry: bool = validate_symmetry
        self.fix: bool = fix


class PoseRangeValidator(PoseFilterBase):
    """Validates that pose feature values are within expected ranges.

    Checks each feature type against its defined range from POSE_FEATURE_RANGES.
    Shows warnings when values are out of range.
    Optionally fixes out-of-range values by clamping them to the valid range.
    """

    def __init__(self, config: PoseRangeValidatorConfig | None = None) -> None:
        super().__init__(config or PoseRangeValidatorConfig())
        self._config: PoseRangeValidatorConfig

    def process(self, pose: Pose) -> Pose:
        """Validate all enabled features and show warnings if values are out of range."""

        if self._config.validate_points:
            point_data = self._validate_feature(pose.point_data, PoseFeatureType.POINTS, "points")
            if point_data is not pose.point_data:
                pose = replace(pose, point_data=point_data)

        if self._config.validate_angles:
            angle_data = self._validate_feature(pose.angle_data, PoseFeatureType.ANGLES, "angles")
            if angle_data is not pose.angle_data:
                pose = replace(pose, angle_data=angle_data)

        if self._config.validate_delta:
            delta_data = self._validate_feature(pose.delta_data, PoseFeatureType.DELTA, "delta")
            if delta_data is not pose.delta_data:
                pose = replace(pose, delta_data=delta_data)

        if self._config.validate_symmetry:
            # Validate but don't modify (it's a cached_property)
            self._validate_feature(pose.symmetry_data, PoseFeatureType.SYMMETRY, "symmetry")

        return pose

    def _validate_feature(self, data: PoseFeatureData, feature_type: PoseFeatureType, feature_name: str) -> PoseFeatureData:
        """Validate feature values are within expected range and show warnings."""
        min_val, max_val = POSE_FEATURE_RANGES[feature_type]

        values: np.ndarray = data.values

        # Check if any finite values are out of range (ignore NaN)
        out_of_range: np.ndarray = np.isfinite(values) & ((values < min_val) | (values > max_val))

        if np.any(out_of_range):
            num_invalid = int(np.sum(out_of_range))
            total_values: int = values.size
            warnings.warn(f"Feature '{feature_name}' has {num_invalid}/{total_values} values out of range [{min_val}, {max_val}]", UserWarning)

            # Fix out-of-range values if enabled
            if self._config.fix:
                fixed_values: np.ndarray = values.copy()
                # Only clamp finite values
                finite_mask: np.ndarray = np.isfinite(fixed_values)
                fixed_values[finite_mask] = np.clip(fixed_values[finite_mask], min_val, max_val)

                # Return the same type as input with fixed values
                return type(data)(values=fixed_values, scores=data.scores)

        return data