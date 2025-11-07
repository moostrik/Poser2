# Standard library imports
from abc import abstractmethod
from dataclasses import replace
from threading import Lock

import numpy as np

# Pose imports
from modules.pose.filter.PoseFilterBase import PoseFilterBase, PoseFilterConfigBase
from modules.pose.Pose import Pose
from modules.pose.filter.prediction.VectorChaseInterpolators import ChaseInterpolator, AngleChaseInterpolator, PointChaseInterpolator
from modules.pose.features import PoseFeatureData, ANGLE_NUM_JOINTS, POSE_NUM_JOINTS, POSE_POINTS_RANGE


class PoseChaseInterpolatorConfig(PoseFilterConfigBase):
    """Configuration for pose chase interpolation with automatic change notification."""

    def __init__(self, input_frequency: float = 30.0, responsiveness: float = 0.2, friction: float = 0.03) -> None:
        super().__init__()
        self.input_frequency: float = input_frequency
        self.responsiveness: float = responsiveness
        self.friction: float = friction


class PoseChaseInterpolatorBase(PoseFilterBase):
    """Base class for pose chase interpolators.

    Handles common interpolation logic using perpetual chase dynamics.
    Subclasses only need to specify:
    - Which interpolator instance to create
    - Which feature to extract/replace from pose
    - How to reconstruct feature data with interpolated values

    The interpolator uses a dual-frequency architecture:
    - process() is called at input frequency (e.g., 30 FPS from pose detection)
    - update() should be called at render frequency (e.g., 60+ FPS for display)

    Note: Interpolated values preserve original confidence scores for valid values,
    but set scores to 0 where interpolations are NaN (insufficient data).
    """

    def __init__(self, config: PoseChaseInterpolatorConfig) -> None:
        self._config: PoseChaseInterpolatorConfig = config
        self._interpolator: ChaseInterpolator
        self._last_pose: Pose | None = None
        self._interpolated_pose: Pose | None = None
        self._pose_lock = Lock()  # Only for pose references
        self._initialize_interpolator()
        self._config.add_listener(self._on_config_changed)

    @property
    def config(self) -> PoseChaseInterpolatorConfig:
        """Access the interpolator's configuration."""
        return self._config

    @abstractmethod
    def _initialize_interpolator(self) -> None:
        """Create the appropriate interpolator instance."""
        pass

    @abstractmethod
    def _get_feature_data(self, pose: Pose) -> PoseFeatureData:
        """Extract the feature data to interpolate from the pose."""
        pass

    @abstractmethod
    def _create_interpolated_data(self, original_data: PoseFeatureData, interpolated_values: np.ndarray) -> PoseFeatureData:
        """Create new feature data with interpolated values."""
        pass

    @abstractmethod
    def _replace_feature_data(self, pose: Pose, new_data: PoseFeatureData) -> Pose:
        """Create new pose with replaced feature data."""
        pass

    def process(self, pose: Pose) -> Pose:
        """Set target from pose. Call at input frequency (e.g., 30 FPS)."""
        with self._pose_lock:
            self._last_pose = pose
        # No lock needed for interpolator - it's thread-safe internally (numpy operations)
        feature_data = self._get_feature_data(pose)
        self._interpolator.set_target(feature_data.values)
        # return original pose unmodified
        return pose

    def update(self, current_time: float | None = None) -> Pose:
        """Update and return interpolated pose. Call at render frequency (e.g., 60+ FPS)."""
        # Lock only for reading _last_pose reference
        with self._pose_lock:
            if self._last_pose is None:
                raise RuntimeError("No pose has been processed yet. Call process() first.")
            last_pose = self._last_pose  # Copy reference

        # No lock needed - interpolator update is safe
        self._interpolator.update(current_time)

        feature_data = self._get_feature_data(last_pose)
        interpolated_values: np.ndarray = self._interpolator.value
        interpolated_data = self._create_interpolated_data(feature_data, interpolated_values)
        interpolated_pose: Pose = self._replace_feature_data(last_pose, interpolated_data)

        # Lock only for writing _interpolated_pose reference
        with self._pose_lock:
            self._interpolated_pose = interpolated_pose

        return interpolated_pose

    def get_interpolated_pose(self) -> Pose:
        """Get a pose with the current interpolated values."""
        with self._pose_lock:
            if self._interpolated_pose is None:
                raise RuntimeError("No interpolated pose available. Call update() first.")
            return self._interpolated_pose

    def reset(self) -> None:
        """Reset the interpolator's internal state."""
        with self._pose_lock:
            self._last_pose = None
            self._interpolated_pose = None
        self._interpolator.reset()  # No lock needed - internal state

    def _on_config_changed(self) -> None:
        """Handle configuration changes by updating interpolator parameters."""
        # No lock needed - config changes are rare and property setters are atomic
        self._interpolator.input_frequency = self._config.input_frequency
        self._interpolator.responsiveness = self._config.responsiveness
        self._interpolator.friction = self._config.friction


class PoseAngleChaseInterpolator(PoseChaseInterpolatorBase):
    """Chase interpolates angle data using perpetual chase dynamics.

    Uses AngleChaseInterpolator which handles circular wrapping of angle values
    and uses shortest angular path for velocity calculations.
    """

    def _initialize_interpolator(self) -> None:
        self._interpolator = AngleChaseInterpolator(
            vector_size=ANGLE_NUM_JOINTS,
            input_frequency=self._config.input_frequency,
            responsiveness=self._config.responsiveness,
            friction=self._config.friction
        )

    def _get_feature_data(self, pose: Pose) -> PoseFeatureData:
        return pose.angle_data

    def _create_interpolated_data(self, original_data: PoseFeatureData, interpolated_values: np.ndarray) -> PoseFeatureData:
        """Create angle data with interpolated values and adjusted scores.

        Sets scores to 0 where interpolations are NaN, preserves original scores otherwise.
        """
        has_nan: np.ndarray = np.isnan(interpolated_values)
        interpolated_scores: np.ndarray = np.where(has_nan, 0.0, original_data.scores).astype(np.float32)
        return type(original_data)(values=interpolated_values, scores=interpolated_scores)

    def _replace_feature_data(self, pose: Pose, new_data: PoseFeatureData) -> Pose:
        return replace(pose, angle_data=new_data)


class PosePointChaseInterpolator(PoseChaseInterpolatorBase):
    """Chase interpolates point data using perpetual chase dynamics.

    Uses PointChaseInterpolator which clamps coordinates to [0, 1] range and handles 2D data.
    """

    def _initialize_interpolator(self) -> None:
        self._interpolator = PointChaseInterpolator(
            num_points=POSE_NUM_JOINTS,
            input_frequency=self._config.input_frequency,
            responsiveness=self._config.responsiveness,
            friction=self._config.friction,
            clamp_range=POSE_POINTS_RANGE
        )

    def _get_feature_data(self, pose: Pose) -> PoseFeatureData:
        return pose.point_data

    def _create_interpolated_data(self, original_data: PoseFeatureData, interpolated_values: np.ndarray) -> PoseFeatureData:
        """Create point data with interpolated values and adjusted scores.

        Checks if ANY coordinate (x or y) is NaN per joint.
        Sets scores to 0 for joints with NaN interpolations, preserves original scores otherwise.
        """
        has_nan: np.ndarray = np.any(np.isnan(interpolated_values), axis=-1)
        interpolated_scores: np.ndarray = np.where(has_nan, 0.0, original_data.scores).astype(np.float32)
        return type(original_data)(values=interpolated_values, scores=interpolated_scores)

    def _replace_feature_data(self, pose: Pose, new_data: PoseFeatureData) -> Pose:
        return replace(pose, point_data=new_data)


class PoseDeltaChaseInterpolator(PoseChaseInterpolatorBase):
    """Chase interpolates delta data using perpetual chase dynamics.

    Uses AngleChaseInterpolator since delta represents angle changes (circular values).
    """

    def _initialize_interpolator(self) -> None:
        self._interpolator = AngleChaseInterpolator(
            vector_size=ANGLE_NUM_JOINTS,
            input_frequency=self._config.input_frequency,
            responsiveness=self._config.responsiveness,
            friction=self._config.friction
        )

    def _get_feature_data(self, pose: Pose) -> PoseFeatureData:
        return pose.delta_data

    def _create_interpolated_data(self, original_data: PoseFeatureData, interpolated_values: np.ndarray) -> PoseFeatureData:
        """Create delta data with interpolated values and adjusted scores.

        Sets scores to 0 where interpolations are NaN, preserves original scores otherwise.
        """
        has_nan: np.ndarray = np.isnan(interpolated_values)
        interpolated_scores: np.ndarray = np.where(has_nan, 0.0, original_data.scores).astype(np.float32)
        return type(original_data)(values=interpolated_values, scores=interpolated_scores)

    def _replace_feature_data(self, pose: Pose, new_data: PoseFeatureData) -> Pose:
        return replace(pose, delta_data=new_data)