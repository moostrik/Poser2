# Standard library imports
import numpy as np

# Pose imports
from ...features import Angles, AngleVelocity, AngleSymmetry, BBox, Points2D, BaseFeature
from ..nodes import FilterNode
from ...frame import Frame, replace
from modules.settings import BaseSettings, Field


class DualConfidenceFilterSettings(BaseSettings):
    """Configuration for hysteresis confidence filtering."""
    threshold_low:  Field[float] = Field(0.3)
    threshold_high: Field[float] = Field(0.5)
    rescale_scores: Field[bool]  = Field(True)


class DualConfidenceFilter(FilterNode):
    """Hysteresis-based confidence filter for pose features.

    Implements dual-threshold filtering to prevent flickering:
    - Values turn ON when score >= threshold_high
    - Values turn OFF when score < threshold_low
    - Once on, values stay on until they drop below threshold_low

    This creates a "sticky" behavior that provides stability when scores
    fluctuate around a threshold boundary.

    Args:
        settings: Hysteresis filter configuration
        feature_type: Feature type specifying which feature to filter

    Example:
        settings = DualConfidenceFilterSettings(threshold_low=0.3, threshold_high=0.5)
        filter = DualConfidenceFilter(settings, Points2D)
    """

    def __init__(self, settings: DualConfidenceFilterSettings, feature_type: type[BaseFeature]):
        self.settings = settings
        self._feature_type = feature_type
        # State: tracks which elements are currently "on" (visible)
        n_elements = feature_type.length()
        self._state_on: np.ndarray = np.zeros(n_elements, dtype=bool)

    def reset(self) -> None:
        """Reset filter state. Called when pose track is lost."""
        n_elements = len(self._state_on)
        self._state_on = np.zeros(n_elements, dtype=bool)

    def process(self, pose: Frame) -> Frame:
        """Filter values using hysteresis thresholds."""
        feature_data = pose[self._feature_type]

        # Skip if no valid values
        if feature_data.valid_count == 0:
            self.reset()
            return pose

        values: np.ndarray = feature_data.values
        scores: np.ndarray = feature_data.scores

        # Hysteresis logic per element:
        # - Turn ON if score >= threshold_high
        # - Turn OFF if score < threshold_low
        # - Otherwise maintain current state
        turn_on = scores >= self.settings.threshold_high
        turn_off = scores < self.settings.threshold_low

        # Update state: turn on takes priority, then turn off, else maintain
        self._state_on = np.where(turn_on, True, np.where(turn_off, False, self._state_on))

        # Apply the hysteresis mask
        # For 2D data (points), broadcast mask to match shape (17,) -> (17, 2)
        if values.ndim > 1:
            filtered_mask = self._state_on[:, np.newaxis]
        else:
            filtered_mask = self._state_on

        filtered_values: np.ndarray = np.where(filtered_mask, values, np.nan)

        # Handle scores based on rescale_scores setting
        if self.settings.rescale_scores:
            # Rescale visible scores to [0, 1] range based on high threshold
            # Division by (1 - threshold_high) remaps [threshold_high, 1.0] -> [0, 1.0]
            filtered_scores: np.ndarray = np.where(
                self._state_on,
                (scores - self.settings.threshold_low) / (1.0 - self.settings.threshold_low),
                0.0
            ).astype(np.float32)
        else:
            # Keep original scores for visible values, set to 0 for hidden values
            filtered_scores: np.ndarray = np.where(self._state_on, scores, 0.0).astype(np.float32)

        # Create new feature data with filtered values
        filtered_data = type(feature_data)(values=filtered_values, scores=filtered_scores)

        # Return new pose with filtered feature
        return replace(pose, {self._feature_type: filtered_data})


# Convenience classes for hysteresis filtering
class BBoxDualConfidenceFilter(DualConfidenceFilter):
    def __init__(self, settings:DualConfidenceFilterSettings) -> None:
        super().__init__(settings,BBox)


class PointDualConfidenceFilter(DualConfidenceFilter):
    def __init__(self, settings:DualConfidenceFilterSettings) -> None:
        super().__init__(settings,Points2D)


class AngleDualConfidenceFilter(DualConfidenceFilter):
    def __init__(self, settings:DualConfidenceFilterSettings) -> None:
        super().__init__(settings,Angles)


class AngleVelocityDualConfidenceFilter(DualConfidenceFilter):
    def __init__(self, settings:DualConfidenceFilterSettings) -> None:
        super().__init__(settings,AngleVelocity)


class AngleSymmetryDualConfidenceFilter(DualConfidenceFilter):
    def __init__(self, settings:DualConfidenceFilterSettings) -> None:
        super().__init__(settings,AngleSymmetry)