"""Pose validators for data integrity checks."""

from threading import Lock
from modules.pose.features.base import BaseFeature

from modules.pose.nodes.Nodes import FilterNode, NodeConfigBase
from modules.pose.Frame import Frame, FrameField


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
        pose_field: PoseField enum value indicating which feature to validate

    Example:
        validator = FeatureValidator(config, PoseField.angles)
        validator = FeatureValidator(config, PoseField.points)
    """

    def __init__(self, config: ValidatorConfig, pose_field: FrameField) -> None:
        if not issubclass(pose_field.get_type(), BaseFeature):
            raise ValueError(f"PoseField '{pose_field.value}' is not a feature field")

        self._config = config
        self._pose_field = pose_field
        self._feature_class = pose_field.get_type()

        self._config_lock: Lock = Lock()
        self._check_ranges: bool = config.check_ranges
        self._name: str = config.name
        self._config.add_listener(self.on_config_changed)

    @property
    def config(self) -> ValidatorConfig:
        return self._config

    def process(self, pose: Frame) -> Frame:
        """Validate feature data and print any errors."""
        feature_data = pose.get_feature(self._pose_field)

        with self._config_lock:
            check_ranges: bool = self._check_ranges
            name: str = self._name

        # Validate using feature's own validation method
        is_valid, error_message = feature_data.validate(check_ranges=check_ranges)

        if not is_valid:
            print(f"{name} validation error in '{self._pose_field.name}' of pose {pose.track_id}: {error_message}")

        # Always return original pose (no fixing, just validation)
        return pose

    def on_config_changed(self) -> None:
        """Handle configuration changes."""
        with self._config_lock:
            self._check_ranges = self._config.check_ranges
            self._name = self._config.name


# Convenience classes
class BBoxValidator(FeatureValidator):
    """Validates bounding box feature data integrity."""
    def __init__(self, config: ValidatorConfig) -> None:
        super().__init__(config, FrameField.bbox)


class PointValidator(FeatureValidator):
    """Validates point feature data integrity."""
    def __init__(self, config: ValidatorConfig) -> None:
        super().__init__(config, FrameField.points)


class AngleValidator(FeatureValidator):
    """Validates angle feature data integrity."""
    def __init__(self, config: ValidatorConfig) -> None:
        super().__init__(config, FrameField.angles)


class AngleVelValidator(FeatureValidator):
    """Validates angle velocity feature data integrity."""
    def __init__(self, config: ValidatorConfig) -> None:
        super().__init__(config, FrameField.angle_vel)


class AngleSymValidator(FeatureValidator):
    """Validates symmetry feature data integrity."""
    def __init__(self, config: ValidatorConfig) -> None:
        super().__init__(config, FrameField.angle_sym)


class PoseValidator(FilterNode):
    """Validates all pose features for data integrity.

    Validates points, angles, angle velocities, bbox, and symmetry features.
    For independent control of each feature, use individual validators.
    """

    def __init__(self, config: ValidatorConfig) -> None:
        self._config = config

        # Create individual validators for each feature
        self._angle_validator = AngleValidator(config)
        self._point_validator = PointValidator(config)
        self._angle_vel_validator = AngleVelValidator(config)
        self._bbox_validator = BBoxValidator(config)
        self._symmetry_validator = AngleSymValidator(config)

    @property
    def config(self) -> ValidatorConfig:
        return self._config

    def process(self, pose: Frame) -> Frame:
        """Validate all features."""
        pose = self._angle_validator.process(pose)
        pose = self._point_validator.process(pose)
        pose = self._angle_vel_validator.process(pose)
        pose = self._bbox_validator.process(pose)
        pose = self._symmetry_validator.process(pose)
        return pose