"""Pose validators for data integrity checks."""

from dataclasses import replace

from modules.pose.features import AngleFeature, Point2DFeature, BBoxFeature, SymmetryFeature
from modules.pose.nodes.Nodes import FilterNode, NodeConfigBase
from modules.pose.Pose import Pose


class ValidatorConfig(NodeConfigBase):
    """Configuration for pose validators."""

    def __init__(self, check_ranges: bool = True) -> None:
        """
        Args:
            check_ranges: Whether to validate value ranges (default: True)
        """
        super().__init__()
        self.check_ranges: bool = check_ranges


class FeatureValidator(FilterNode):
    """Generic pose feature validator.

    Validates feature data integrity by calling the feature's validate() method.
    Prints validation errors to console for debugging.

    Args:
        config: Validator configuration
        feature_class: Feature class type (e.g., AngleFeature, Point2DFeature)
        attr_name: Name of the pose attribute to validate

    Example:
        validator = FeatureValidator(config, AngleFeature, "angles")
        validator = FeatureValidator(config, Point2DFeature, "points")
    """

    def __init__(self, config: ValidatorConfig, feature_class: type, attr_name: str):
        self._config = config
        self._feature_class = feature_class
        self._attr_name = attr_name

    @property
    def config(self) -> ValidatorConfig:
        return self._config

    def process(self, pose: Pose) -> Pose:
        """Validate feature data and print any errors."""
        feature_data = getattr(pose, self._attr_name)

        # Validate using feature's own validation method
        is_valid, error_message = feature_data.validate(check_ranges=self._config.check_ranges)

        if not is_valid:
            print(f"Validation error in '{self._attr_name}' of pose {pose.track_id}: {error_message}")

        # Always return original pose (no fixing, just validation)
        return pose


# Convenience classes
class AngleValidator(FeatureValidator):
    """Validates angle feature data integrity."""
    def __init__(self, config: ValidatorConfig) -> None:
        super().__init__(config, AngleFeature, "angles")


class PointValidator(FeatureValidator):
    """Validates point feature data integrity."""
    def __init__(self, config: ValidatorConfig) -> None:
        super().__init__(config, Point2DFeature, "points")


class DeltaValidator(FeatureValidator):
    """Validates delta feature data integrity."""
    def __init__(self, config: ValidatorConfig) -> None:
        super().__init__(config, AngleFeature, "deltas")


class BBoxValidator(FeatureValidator):
    """Validates bounding box feature data integrity."""
    def __init__(self, config: ValidatorConfig) -> None:
        super().__init__(config, BBoxFeature, "bbox")


class SymmetryValidator(FeatureValidator):
    """Validates symmetry feature data integrity."""
    def __init__(self, config: ValidatorConfig) -> None:
        super().__init__(config, SymmetryFeature, "symmetry")


class PoseValidator(FilterNode):
    """Validates all pose features for data integrity.

    Validates points, angles, deltas, and symmetry features.
    For independent control of each feature, use individual validators.
    """

    def __init__(self, config: ValidatorConfig) -> None:
        self._config = config

        # Create individual validators for each feature
        self._angle_validator = AngleValidator(config)
        self._point_validator = PointValidator(config)
        self._delta_validator = DeltaValidator(config)
        self._bbox_validator = BBoxValidator(config)
        self._symmetry_validator = SymmetryValidator(config)

    @property
    def config(self) -> ValidatorConfig:
        return self._config

    def process(self, pose: Pose) -> Pose:
        """Validate all features."""
        pose = self._angle_validator.process(pose)
        pose = self._point_validator.process(pose)
        pose = self._delta_validator.process(pose)
        pose = self._bbox_validator.process(pose)
        pose = self._symmetry_validator.process(pose)
        return pose