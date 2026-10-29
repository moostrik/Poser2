"""Base class for pose feature interpolators with common infrastructure.

Provides shared functionality for both chase and lerp interpolators:
- Feature class registry and validation
- Thread-safe pose management
- Settings listener management
- Interpolated data creation

Thread Safety:
--------------
Uses a lock to serialize all access, preventing concurrent set_target() and update() calls.
"""

# Standard library imports
from abc import ABC, abstractmethod
from enum import IntEnum
from threading import Lock
from typing import TypeVar, Generic, Protocol
from collections import defaultdict

# Third-party imports
import numpy as np

# Pose imports
from ...features import BaseFeature
from ..nodes import InterpolatorNode
from ...frame import Frame, replace
from modules.settings import BaseSettings, Field


class TickStep(IntEnum):
    """How far an interpolator advances on a tick."""
    FIXED    = 0     # one output interval, whenever the tick comes: steady on a regular output clock
    MEASURED = 1     # the time since the previous tick: a late tick moves further, a stall all the way
    CAPPED   = 2     # the time since the previous tick, at most one input frame: a stall moves one frame


class InterpolatorSettingsBase(BaseSettings):
    """Base config for interpolators: the input rate (the horizon the chase aims at), the output rate of a fixed
    step and the step mode."""
    input_frequency:  Field[float] = Field(30.0, access=Field.INIT)
    output_frequency: Field[float] = Field(60.0)
    step: Field[TickStep] = Field(TickStep.FIXED, description="A tick's step: one output interval, the time since the previous tick, or that capped at one input frame")


# Generic type variable for settings
SettingsType = TypeVar('SettingsType', bound=InterpolatorSettingsBase)


class InterpolatorProtocol(Protocol):
    """Protocol defining the interface for interpolator classes."""

    def set_target(self, values: np.ndarray) -> None:
        """Set new target values."""
        ...

    def update(self, dt: float) -> None:
        """Update interpolated values by time delta."""
        ...

    @property
    def value(self) -> np.ndarray:
        """Get current interpolated values."""
        ...

    def reset(self) -> None:
        """Reset interpolator state."""
        ...


class FeatureInterpolatorBase(InterpolatorNode, ABC, Generic[SettingsType]):
    """Base class for pose feature interpolators."""

    # Registry mapping feature types to interpolator classes
    # Subclasses must override this with their specific interpolator types
    _INTERP_MAP: defaultdict[type[BaseFeature], type]

    def __init__(self, settings: SettingsType, feature_type: type[BaseFeature]):
        """Initialize the feature interpolator."""
        self.settings: SettingsType = settings
        self._feature_type: type[BaseFeature] = feature_type
        self._lock: Lock = Lock()
        self._last_pose: Frame | None = None
        self._last_now: float | None = None                           # the previous tick's stamp; None before the first
        self._max_dt: float = 1.0 / settings.input_frequency          # one input frame: the cap of a CAPPED step
        self._step: TickStep = TickStep.FIXED
        self._output_interval: float = 1.0 / settings.output_frequency
        self._read_step()
        self._interpolator: InterpolatorProtocol = self._create_interpolator()
        self.settings.bind_all(self._on_settings_changed)

    def __del__(self):
        """Cleanup settings listener to prevent memory leaks."""
        try:
            self.settings.unbind_all(self._on_settings_changed)
        except (AttributeError, ValueError):
            pass

    @abstractmethod
    def _create_interpolator(self) -> InterpolatorProtocol:
        """Create the underlying interpolator instance."""
        pass

    def _on_settings_changed(self, _=None) -> None:
        """Handle configuration changes by updating interpolator parameters; subclasses extend it."""
        with self._lock:
            self._read_step()

    def _read_step(self) -> None:
        """Cache the step mode and the fixed step's interval; called under the lock after construction."""
        self._step = TickStep(int(self.settings.step))
        self._output_interval = 1.0 / self.settings.output_frequency

    def _dt(self, now: float) -> float:
        """The step of the tick ``now`` by the step mode; records the tick for the next."""
        previous, self._last_now = self._last_now, now
        if self._step == TickStep.FIXED:
            return self._output_interval
        if previous is None:                                           # the first tick after a set or a reset
            return 0.0                                                 # moves nothing: the target was taken at once
        dt = now - previous
        return min(dt, self._max_dt) if self._step == TickStep.CAPPED else dt

    def is_ready(self) -> bool:
        """Check if the generator is ready to produce a pose."""
        return self._last_pose is not None

    @property
    def feature_type(self) -> type[BaseFeature]:
        """Return the feature type this interpolator processes."""
        return self._feature_type

    def set(self, pose: Frame) -> None:
        """Set target from pose. Call at input frequency (e.g., 30 FPS)."""
        feature_data = pose[self._feature_type]

        with self._lock:
            self._interpolator.set_target(feature_data.values)
            self._last_pose = pose

    def update(self, now: float) -> Frame | None:
        """The interpolated pose at the tick ``now``, advanced by the step of the settings' ``step`` mode: one output
        interval (FIXED), the time since the previous tick (MEASURED) or that capped at one input frame (CAPPED).
        Call on the output clock.

        Returns:
            Interpolated pose, or None if set() has not been called yet.
        """
        with self._lock:
            if self._last_pose is None:
                return None
            last_pose = self._last_pose

            self._interpolator.update(self._dt(now))
            interpolated_values: np.ndarray = self._interpolator.value

        feature_data = last_pose[self._feature_type]
        interpolated_data = self._create_interpolated_data(feature_data, interpolated_values)
        return replace(last_pose, {self._feature_type: interpolated_data})

    def _create_interpolated_data(self, original_data: BaseFeature,
                                 interpolated_values: np.ndarray) -> BaseFeature:
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
            self._last_now = None
            self._interpolator.reset()