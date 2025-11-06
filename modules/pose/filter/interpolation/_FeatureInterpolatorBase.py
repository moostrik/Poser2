# Standard library imports
from abc import abstractmethod
from typing import Any, Generic, TypeVar

# Local application imports
from modules.pose.filter.interpolation.InterpolatorConfig import InterpolatorConfig


# Generic type for any feature (PoseAngles, PoseMeasurements, etc.)
TFeature = TypeVar('TFeature')


class FeatureInterpolatorBase(Generic[TFeature]):
    """Base class for feature interpolation with separated input and output rates.

    Separates input sampling (at camera FPS) from output generation (at display/processing FPS):
    - add_feature(): Feeds input feature sample to interpolator (called at input_rate)
    - update(): Generates interpolated feature (called at output_rate)
    - reset(): Reinitializes filter state (called externally when needed)

    Handles:
    - Filter state management
    - Dual-rate processing (input vs output)

    Subclasses implement:
    - Filter initialization
    - Sample addition logic for specific feature types
    - Interpolation/update logic for generating output
    """

    def __init__(self, config: InterpolatorConfig) -> None:
        super().__init__()
        self._config: InterpolatorConfig = config
        # Initialize state immediately
        self._state: Any = self._create_state()

        # Register for config change notifications
        config.add_listener(self._on_config_changed)

    def add_feature(self, feature: TFeature) -> None:
        """Add feature sample to interpolator (called at input rate)."""
        self._add_sample(feature)

    def update(self, current_time: float | None = None) -> TFeature:
        """Generate interpolated feature at current time (called at output rate)."""
        return self._interpolate(current_time)

    def reset(self) -> None:
        """Reinitialize filter state (call when tracklet is lost or reset needed)."""
        self._state = self._create_state()

    @abstractmethod
    def _create_state(self) -> Any:
        """Create initial filter state."""
        pass

    @abstractmethod
    def _add_sample(self, feature: TFeature) -> None:
        """Add feature sample to interpolator. """
        pass

    @abstractmethod
    def _interpolate(self, current_time: float | None) -> TFeature:
        """Generate interpolated feature at current time."""
        pass

    @abstractmethod
    def _on_config_changed(self) -> None:
        """Called when config changes. Override to apply new config values."""
        pass  # Default: no action needed