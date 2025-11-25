"""Base class for pose feature interpolators with common infrastructure.

Provides shared functionality for both chase and lerp interpolators:
- Feature class registry and validation
- Thread-safe pose management
- Config listener management
- Interpolated data creation

Thread Safety:
--------------
Uses a lock to serialize all access, preventing concurrent set_target() and update() calls.
"""

# Standard library imports
from abc import ABC, abstractmethod
from dataclasses import replace
from threading import Lock
from typing import TypeVar, Generic, Protocol
from collections import defaultdict

# Third-party imports
import numpy as np

# Pose imports
from modules.pose.features import PoseFeatureType, Angles, BBox, Points2D, AngleSymmetry
from modules.pose.nodes.Nodes import InterpolatorNode, NodeConfigBase
from modules.pose.Frame import Frame, FrameField

# Generic type variable for config
ConfigType = TypeVar('ConfigType', bound=NodeConfigBase)


class InterpolatorProtocol(Protocol):
    """Protocol defining the interface for interpolator classes."""

    def set_target(self, values: np.ndarray) -> None:
        """Set new target values."""
        ...

    def update(self, current_time: float | None = None) -> None:
        """Update interpolated values."""
        ...

    @property
    def value(self) -> np.ndarray:
        """Get current interpolated values."""
        ...

    def reset(self) -> None:
        """Reset interpolator state."""
        ...


class FeatureInterpolatorBase(InterpolatorNode, ABC, Generic[ConfigType]):
    """Base class for pose feature interpolators."""

    # Registry mapping feature classes to interpolator classes
    # Subclasses must override this with their specific interpolator types
    _INTERP_MAP: defaultdict[FrameField, type]

    def __init__(self, config: ConfigType, pose_field: FrameField):
        """Initialize the feature interpolator."""
        self._config: ConfigType = config
        self._pose_field: FrameField = pose_field
        self._lock: Lock = Lock()
        self._last_pose: Frame | None = None
        self._interpolator: InterpolatorProtocol = self._create_interpolator()
        self._config.add_listener(self._on_config_changed)

    def __del__(self):
        """Cleanup config listener to prevent memory leaks."""
        try:
            self._config.remove_listener(self._on_config_changed)
        except (AttributeError, ValueError):
            pass

    @abstractmethod
    def _create_interpolator(self) -> InterpolatorProtocol:
        """Create the underlying interpolator instance."""
        pass

    @abstractmethod
    def _on_config_changed(self) -> None:
        """Handle configuration changes by updating interpolator parameters."""
        pass

    def is_ready(self) -> bool:
        """Check if the generator is ready to produce a pose."""
        return self._last_pose is not None

    @property
    def config(self) -> ConfigType:
        """Get the configuration object."""
        return self._config

    @property
    def attr_name(self) -> str:
        """Return the attribute name this interpolator processes."""
        return self._pose_field.name

    def submit(self, pose: Frame) -> None:
        """Set target from pose. Call at input frequency (e.g., 30 FPS)."""
        feature_data = pose.get_feature(self._pose_field)

        with self._lock:
            self._interpolator.set_target(feature_data.values)
            self._last_pose = pose

    def update(self, time_stamp: float | None = None) -> Frame | None:
        """Update and return interpolated pose. Call at render frequency (e.g., 60+ FPS).

        Returns:
            Interpolated pose, or None if submit() has not been called yet.
        """
        with self._lock:
            if self._last_pose is None:
                return None
            last_pose = self._last_pose

            self._interpolator.update(time_stamp)
            interpolated_values: np.ndarray = self._interpolator.value

        feature_data = last_pose.get_feature(self._pose_field)
        interpolated_data = self._create_interpolated_data(feature_data, interpolated_values)
        return replace(last_pose, **{self._pose_field.name: interpolated_data}, time_stamp=time_stamp)

    def _create_interpolated_data(self, original_data: PoseFeatureType,
                                 interpolated_values: np.ndarray) -> PoseFeatureType:
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