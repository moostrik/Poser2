"""Pose validators for data integrity checks."""

from threading import Lock
from ...features import BBox, Points2D, Angles, AngleVelocity, AngleSymmetry, BaseFeature

from ..Nodes import FilterNode
from ...frame import Frame
from modules.settings import BaseSettings, Field

import logging
logger = logging.getLogger(__name__)


class ValidatorSettings(BaseSettings):
    """Configuration for pose validators."""
    check_ranges: Field[bool] = Field(True)
    name:         Field[str]  = Field('default')


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

    def __init__(self, config: ValidatorSettings, feature_type: type[BaseFeature]) -> None:
        self._config = config
        self._feature_type: type[BaseFeature] = feature_type

        self._config_lock: Lock = Lock()
        self._check_ranges: bool = config.check_ranges
        self._name: str = config.name
        self._config.add_listener(self.on_config_changed)

    @property
    def config(self) -> ValidatorSettings:
        return self._config

    def process(self, pose: Frame) -> Frame:
        """Validate feature data and print any errors."""
        feature_data = pose[self._feature_type]

        with self._config_lock:
            check_ranges: bool = self._check_ranges
            name: str = self._name

        # Validate using feature's own validation method
        is_valid, error_message = feature_data.validate(check_ranges=check_ranges)

        if not is_valid:
            logger.error(f"{name} validation error in '{self._feature_type.__name__}' of pose {pose.track_id}: {error_message}")

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
    def __init__(self, config: ValidatorSettings) -> None:
        super().__init__(config, BBox)


class PointValidator(FeatureValidator):
    """Validates point feature data integrity."""
    def __init__(self, config: ValidatorSettings) -> None:
        super().__init__(config, Points2D)


class AngleValidator(FeatureValidator):
    """Validates angle feature data integrity."""
    def __init__(self, config: ValidatorSettings) -> None:
        super().__init__(config, Angles)


class AngleVelValidator(FeatureValidator):
    """Validates angle velocity feature data integrity."""
    def __init__(self, config: ValidatorSettings) -> None:
        super().__init__(config, AngleVelocity)


class AngleSymValidator(FeatureValidator):
    """Validates symmetry feature data integrity."""
    def __init__(self, config: ValidatorSettings) -> None:
        super().__init__(config, AngleSymmetry)


class PoseValidator(FilterNode):
    """Validates all pose features for data integrity.

    Validates points, angles, angle velocities, bbox, and symmetry features.
    For independent control of each feature, use individual validators.
    """

    def __init__(self, config: ValidatorSettings) -> None:
        self._config = config

        # Create individual validators for each feature
        self._angle_validator = AngleValidator(config)
        self._point_validator = PointValidator(config)
        self._angle_vel_validator = AngleVelValidator(config)
        self._bbox_validator = BBoxValidator(config)
        self._symmetry_validator = AngleSymValidator(config)

    @property
    def config(self) -> ValidatorSettings:
        return self._config

    def process(self, pose: Frame) -> Frame:
        """Validate all features."""
        pose = self._angle_validator.process(pose)
        pose = self._point_validator.process(pose)
        pose = self._angle_vel_validator.process(pose)
        pose = self._bbox_validator.process(pose)
        pose = self._symmetry_validator.process(pose)
        return pose