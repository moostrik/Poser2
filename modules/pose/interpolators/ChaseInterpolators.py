"""Pose chase interpolation filters for smooth motion interpolation.

Provides perpetual chase interpolation for angles, points, and deltas with
proper handling of circular values and coordinate clamping. Uses dual-frequency
architecture: process() at input rate, update() at render rate.

Thread Safety:
--------------
Designed for multi-threaded operation:
- Input thread: Calls process() at ~30 FPS
- Render thread: Calls update() at 60+ FPS

VectorChaseInterpolator classes are NOT thread-safe. PoseChaseInterpolatorBase
serializes all access using a lock to prevent concurrent set_target() and update() calls
and makes sure the last_pose always corresponds to interpolator's last set target.
"""

# Standard library imports
from abc import abstractmethod
from dataclasses import replace
from threading import Lock

import numpy as np

# Pose imports
from modules.pose.Nodes import FilterNode, NodeConfigBase
from modules.pose.Pose import Pose
from modules.pose.interpolators.algorithms.VectorChase import Chase, VectorChase, AngleChase, PointChase
from modules.pose.features import PoseFeatureData, ANGLE_NUM_LANDMARKS, POINT_NUM_LANDMARKS, POINT2D_COORD_RANGE


class ChaseInterpolatorConfig(NodeConfigBase):
    """Configuration for pose chase interpolation with automatic change notification."""

    def __init__(self, input_frequency: float = 30.0, responsiveness: float = 0.2, friction: float = 0.03) -> None:
        super().__init__()
        self.input_frequency: float = input_frequency
        self.responsiveness: float = responsiveness
        self.friction: float = friction


class ChaseInterpolatorBase(FilterNode):
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

    def __init__(self, config: ChaseInterpolatorConfig) -> None:
        self._config: ChaseInterpolatorConfig = config
        self._lock: Lock = Lock()
        self._last_pose: Pose | None = None
        self._interpolator: Chase
        self._initialize_interpolator()
        self._config.add_listener(self._on_config_changed)

    @property
    def config(self) -> ChaseInterpolatorConfig:
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
        feature_data = self._get_feature_data(pose)

        # Atomic block: update both together
        with self._lock:
            self._interpolator.set_target(feature_data.values)
            self._last_pose = pose

        return pose

    def update(self, current_time: float | None = None) -> Pose | None:
        """Update and return interpolated pose. Call at render frequency (e.g., 60+ FPS).
        Returns None if process() has not been called yet."""
        # Lock only for reading _last_pose reference
        with self._lock:
            if self._last_pose is None:
                return None
            last_pose: Pose = self._last_pose

            self._interpolator.update(current_time)
            interpolated_values: np.ndarray = self._interpolator.value

        feature_data = self._get_feature_data(last_pose)
        interpolated_data = self._create_interpolated_data(feature_data, interpolated_values)
        interpolated_pose: Pose = self._replace_feature_data(last_pose, interpolated_data)

        return interpolated_pose

    def reset(self) -> None:
        """Reset the interpolator's internal state."""
        with self._lock:
            self._last_pose = None
            self._interpolated_pose = None
            self._interpolator.reset()  # No lock needed - internal state

    def _on_config_changed(self) -> None:
        """Handle configuration changes by updating interpolator parameters."""
        with self._lock:
            self._interpolator.input_frequency = self._config.input_frequency
            self._interpolator.responsiveness = self._config.responsiveness
            self._interpolator.friction = self._config.friction


class AngleChaseInterpolator(ChaseInterpolatorBase):
    """Chase interpolates angle data using perpetual chase dynamics.

    Uses AngleChaseInterpolator which handles circular wrapping of angle values
    and uses shortest angular path for velocity calculations.
    """

    def _initialize_interpolator(self) -> None:
        self._interpolator = AngleChase(
            vector_size=ANGLE_NUM_LANDMARKS,
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


class PointChaseInterpolator(ChaseInterpolatorBase):
    """Chase interpolates point data using perpetual chase dynamics.

    Uses PointChaseInterpolator which clamps coordinates to [0, 1] range and handles 2D data.
    """

    def _initialize_interpolator(self) -> None:
        self._interpolator = PointChase(
            num_points=POINT_NUM_LANDMARKS,
            input_frequency=self._config.input_frequency,
            responsiveness=self._config.responsiveness,
            friction=self._config.friction,
            clamp_range=POINT2D_COORD_RANGE
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


class DeltaChaseInterpolator(ChaseInterpolatorBase):
    """Chase interpolates delta data using perpetual chase dynamics.

    Uses AngleChaseInterpolator since delta represents angle changes (circular values).
    """

    def _initialize_interpolator(self) -> None:
        self._interpolator = AngleChase(
            vector_size=ANGLE_NUM_LANDMARKS,
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


class PoseChaseInterpolator(FilterNode):
    """Chase interpolates all pose features (angles, points, and deltas)."""

    def __init__(self, config: ChaseInterpolatorConfig) -> None:
        self._config: ChaseInterpolatorConfig = config
        self._cached_pose: Pose | None = None

        self._angle_interpolator = AngleChaseInterpolator(config)
        self._point_interpolator = PointChaseInterpolator(config)
        self._delta_interpolator = DeltaChaseInterpolator(config)

    @property
    def config(self) -> ChaseInterpolatorConfig:
        return self._config

    def process(self, pose: Pose) -> Pose:
        """Set target from pose. Call at input frequency (e.g., 30 FPS)."""
        self._angle_interpolator.process(pose)
        self._point_interpolator.process(pose)
        self._delta_interpolator.process(pose)
        return pose

    def update(self, current_time: float | None = None) -> Pose | None:
        """Update and return interpolated pose, or None if not ready yet.

        Returns None if process() has not been called yet.
        """

        # Update all features (each caches internally)
        angle_pose: Pose | None = self._angle_interpolator.update(current_time)
        point_pose: Pose | None = self._point_interpolator.update(current_time)
        delta_pose: Pose | None = self._delta_interpolator.update(current_time)

        if angle_pose is None or point_pose is None or delta_pose is None:
            return None

        # Combine all interpolated features
        combined: Pose = replace(
            angle_pose,
            point_data=point_pose.point_data,
            delta_data=delta_pose.delta_data
        )

        # Cache combined result
        self._cached_pose = combined

        return combined

    def reset(self) -> None:
        """Reset all interpolators' internal state."""
        self._angle_interpolator.reset()
        self._point_interpolator.reset()
        self._delta_interpolator.reset()

        self._cached_pose = None