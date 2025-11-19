"""Pose chase interpolation filters for smooth motion interpolation.

Provides perpetual chase interpolation for angles, points, and deltas with
proper handling of circular values and coordinate clamping. Uses dual-frequency
architecture: process() at input rate, update() at render rate.

Thread Safety:
--------------
Designed for multi-threaded operation:
- Input thread: Calls process() at ~30 FPS
- Render thread: Calls update() at 60+ FPS

VectorChaseInterpolator classes are NOT thread-safe. FeatureChaseInterpolator
serializes all access using a lock to prevent concurrent set_target() and update() calls
and makes sure the last_pose always corresponds to interpolator's last set target.
"""

# Standard library imports
from dataclasses import replace
from threading import Lock
from time import monotonic

# Third-party imports
import numpy as np

# Pose imports
from modules.pose.features import PoseFeature, Angles, BBox, Points2D, Symmetry
from modules.pose.nodes._utils.VectorChase import AngleChase, PointChase, VectorChase
from modules.pose.nodes.Nodes import InterpolatorNode, NodeConfigBase
from modules.pose.Pose import Pose


class ChaseInterpolatorConfig(NodeConfigBase):
    """Configuration for pose chase interpolation with automatic change notification."""

    def __init__(self, input_frequency: float = 30.0, responsiveness: float = 0.2, friction: float = 0.03) -> None:
        super().__init__()
        self.input_frequency: float = input_frequency
        self.responsiveness: float = responsiveness
        self.friction: float = friction


class FeatureChaseInterpolator(InterpolatorNode):
    """Generic pose feature chase interpolator."""

    # Registry mapping feature classes to interpolator classes
    INTERPOLATOR_REGISTRY = {
        Angles: AngleChase,
        BBox: VectorChase,
        Points2D: PointChase,
        Symmetry: VectorChase,
    }

    def __init__(self, config: ChaseInterpolatorConfig, feature_class: type, attr_name: str):
        if feature_class not in self.INTERPOLATOR_REGISTRY:
            valid_classes = [cls.__name__ for cls in self.INTERPOLATOR_REGISTRY.keys()]
            raise ValueError(
                f"Unknown feature class '{feature_class.__name__}'. "
                f"Must be one of: {valid_classes}"
            )

        self._config = config
        self._feature_class = feature_class
        self._attr_name = attr_name
        self._lock = Lock()
        self._last_pose: Pose | None = None

        interpolator_cls = self.INTERPOLATOR_REGISTRY[feature_class]
        self._interpolator = interpolator_cls(
            vector_size=len(feature_class.feature_enum()),
            input_frequency=config.input_frequency,
            responsiveness=config.responsiveness,
            friction=config.friction,
            clamp_range=feature_class.default_range()
        )

        self._config.add_listener(self._on_config_changed)

    def __del__(self):
        """Cleanup config listener to prevent memory leaks."""
        try:
            self._config.remove_listener(self._on_config_changed)
        except (AttributeError, ValueError):
            pass

    @property
    def config(self) -> ChaseInterpolatorConfig:
        return self._config

    @property
    def attr_name(self) -> str:
        """Return the attribute name this interpolator processes."""
        return self._attr_name

    def submit(self, pose: Pose) -> None:
        """Set target from pose. Call at input frequency (e.g., 30 FPS)."""
        feature_data = getattr(pose, self._attr_name)

        with self._lock:
            self._interpolator.set_target(feature_data.values)
            self._last_pose = pose

    def update(self, time_stamp: float | None = None) -> Pose | None:
        """Update and return interpolated pose. Call at render frequency (e.g., 60+ FPS).
        Returns None if process() has not been called yet."""

        with self._lock:
            if self._last_pose is None:
                return None
            last_pose = self._last_pose

            if time_stamp is None:
                time_stamp = monotonic()

            self._interpolator.update(time_stamp)
            interpolated_values: np.ndarray = self._interpolator.value

        feature_data = getattr(last_pose, self._attr_name)
        interpolated_data = self._create_interpolated_data(feature_data, interpolated_values)
        return replace(last_pose, **{self._attr_name: interpolated_data}, time_stamp=time_stamp)

    def _create_interpolated_data(self, original_data: PoseFeature, interpolated_values: np.ndarray) -> PoseFeature:
        """Create feature data with interpolated values and adjusted scores."""
        if interpolated_values.ndim > 1:
            has_nan = np.any(np.isnan(interpolated_values), axis=-1)
        else:
            has_nan = np.isnan(interpolated_values)

        interpolated_scores = np.where(has_nan, 0.0, original_data.scores).astype(np.float32)
        return type(original_data)(values=interpolated_values, scores=interpolated_scores)

    def reset(self) -> None:
        """Reset the interpolator's internal state."""
        with self._lock:
            self._last_pose = None
            self._interpolator.reset()

    def _on_config_changed(self) -> None:
        """Handle configuration changes by updating interpolator parameters."""
        with self._lock:
            self._interpolator.input_frequency = self._config.input_frequency
            self._interpolator.responsiveness = self._config.responsiveness
            self._interpolator.friction = self._config.friction


# Convenience classes
class AngleChaseInterpolator(FeatureChaseInterpolator):
    def __init__(self, config: ChaseInterpolatorConfig) -> None:
        super().__init__(config, Angles, "angles")


class BBoxChaseInterpolator(FeatureChaseInterpolator):
    def __init__(self, config: ChaseInterpolatorConfig) -> None:
        super().__init__(config, BBox, "bbox")


class DeltaChaseInterpolator(FeatureChaseInterpolator):
    def __init__(self, config: ChaseInterpolatorConfig) -> None:
        super().__init__(config, Angles, "deltas")


class PointChaseInterpolator(FeatureChaseInterpolator):
    def __init__(self, config: ChaseInterpolatorConfig) -> None:
        super().__init__(config, Points2D, "points")


class SymmetryChaseInterpolator(FeatureChaseInterpolator):
    def __init__(self, config: ChaseInterpolatorConfig) -> None:
        super().__init__(config, Symmetry, "symmetry")


# class PointAngleChaseInterpolator(InterpolatorNode):
#     def __init__(self, config: ChaseInterpolatorConfig):
#         super().__init__()
#         self.point_interpolator = PointChaseInterpolator(config)
#         self.angle_interpolator = AngleChaseInterpolator(config)
#         self._last_pose: Pose | None = None

#     def submit(self, pose: Pose) -> None:
#         self.point_interpolator.submit(pose)
#         self.angle_interpolator.submit(pose)
#         self._last_pose = pose

#     def update(self, current_time: float | None = None) -> Pose | None:
#         if self._last_pose is None:
#             return None

#         # Update both interpolators
#         interpolated_points_pose = self.point_interpolator.update(current_time)
#         interpolated_angles_pose = self.angle_interpolator.update(current_time)

#         # Merge interpolated features into a new pose
#         # (Assumes Pose is a dataclass and can be replaced with new features)
#         return replace(
#             self._last_pose,
#             points=getattr(interpolated_points_pose, "points"),
#             angles=getattr(interpolated_angles_pose, "angles"),
#         )