"""Pose validators for data integrity checks."""

from threading import Lock

from modules.pose.features import Angles, Points2D, BBox, Symmetry
from modules.pose.nodes.Nodes import FilterNode, NodeConfigBase
from modules.pose.Pose import Pose


class ValidatorConfig(NodeConfigBase):
    """Configuration for pose validators."""

    def __init__(self, check_ranges: bool = True, name: str = "default") -> None:
        """
        Args:
            check_ranges: Whether to validate value ranges (default: True)
        """
        super().__init__()
        self.check_ranges: bool = check_ranges
        self.name: str = name


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

        self._config_lock: Lock = Lock()
        self._check_ranges: bool = config.check_ranges
        self._name: str = config.name
        self._config.add_listener(self.on_config_changed)

    @property
    def config(self) -> ValidatorConfig:
        return self._config

    def process(self, pose: Pose) -> Pose:
        """Validate feature data and print any errors."""
        feature_data = getattr(pose, self._attr_name)

        with self._config_lock:
            check_ranges: bool = self._check_ranges
            name: str = self._name

        # Validate using feature's own validation method
        is_valid, error_message = feature_data.validate(check_ranges=check_ranges)

        if not is_valid:
            print(f"{name} validation error in '{self._attr_name}' of pose {pose.track_id}: {error_message}")

        # Always return original pose (no fixing, just validation)
        return pose

    def on_config_changed(self) -> None:
        """Handle configuration changes."""
        with self._config_lock:
            self._check_ranges = self._config.check_ranges
            self._name = self._config.name


# Convenience classes
class AngleValidator(FeatureValidator):
    """Validates angle feature data integrity."""
    def __init__(self, config: ValidatorConfig) -> None:
        super().__init__(config, Angles, "angles")


class PointValidator(FeatureValidator):
    """Validates point feature data integrity."""
    def __init__(self, config: ValidatorConfig) -> None:
        super().__init__(config, Points2D, "points")


class DeltaValidator(FeatureValidator):
    """Validates delta feature data integrity."""
    def __init__(self, config: ValidatorConfig) -> None:
        super().__init__(config, Angles, "deltas")


class BBoxValidator(FeatureValidator):
    """Validates bounding box feature data integrity."""
    def __init__(self, config: ValidatorConfig) -> None:
        super().__init__(config, BBox, "bbox")


class SymmetryValidator(FeatureValidator):
    """Validates symmetry feature data integrity."""
    def __init__(self, config: ValidatorConfig) -> None:
        super().__init__(config, Symmetry, "symmetry")


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