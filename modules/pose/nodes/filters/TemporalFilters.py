"""Temporal stability filters for pose features.

Provides temporal-based filtering where values must be consistently present
for a minimum number of consecutive frames before being considered valid.
This prevents transient detections and flickering.
"""

# Standard library imports
import numpy as np

# Pose imports
from modules.pose.features import Angles, AngleVelocity, AngleSymmetry, BBox, Points2D
from modules.pose.features.base import BaseFeature
from modules.pose.nodes.Nodes import FilterNode
from modules.pose.frame import Frame, replace
from modules.settings import BaseSettings, Field


class TemporalStabilizerSettings(BaseSettings):
    """Configuration for temporal stability filtering."""
    min_consecutive_frames: Field[int] = Field(3)
    max_gap_frames:         Field[int] = Field(0)


class TemporalStabilizer(FilterNode):
    """Temporal stability filter for pose features.

    Tracks per-element validity over time using a consecutive frame counter:
    - Counter increments when value is valid (non-NaN)
    - Counter decrements (with gap tolerance) when value is invalid
    - Element becomes visible when counter >= min_consecutive_frames
    - Element becomes hidden when counter drops to 0

    This creates temporal persistence, filtering out momentary detections
    and preventing rapid flickering.

    Important: Apply confidence/hysteresis filtering BEFORE this filter.
    This filter operates on the valid_mask, not raw scores.

    Args:
        config: Temporal stability filter configuration
        feature_type: Feature type specifying which feature to filter

    Example:
        config = TemporalStabilityConfig(
            min_consecutive_frames=3,
            max_gap_frames=1
        )
        filter = TemporalStabilityFilter(config, Points2D)
    """

    def __init__(self, config: TemporalStabilizerSettings, feature_type: type[BaseFeature]):
        self._config = config
        self._feature_type = feature_type
        # Get number of elements from feature type
        n_elements = feature_type.length()

        # State: tracks consecutive frame count for each element
        # Positive values = consecutive frames valid, 0 = not yet stable
        self._consecutive_count: np.ndarray = np.zeros(n_elements, dtype=np.int32)
        # State: tracks current gap count (frames invalid while in grace period)
        self._gap_count: np.ndarray = np.zeros(n_elements, dtype=np.int32)
        # State: tracks which elements are currently visible
        self._visible: np.ndarray = np.zeros(n_elements, dtype=bool)

    @property
    def config(self) -> TemporalStabilizerSettings:
        return self._config

    def reset(self) -> None:
        """Reset filter state. Called when pose track is lost."""
        n_elements = len(self._consecutive_count)
        self._consecutive_count = np.zeros(n_elements, dtype=np.int32)
        self._gap_count = np.zeros(n_elements, dtype=np.int32)
        self._visible = np.zeros(n_elements, dtype=bool)

    def process(self, pose: Frame) -> Frame:
        """Filter values requiring temporal stability."""
        feature_data = pose[self._feature_type]

        # Skip if no valid values
        if feature_data.valid_count == 0:
            self.reset()
            return pose

        values: np.ndarray = feature_data.values
        scores: np.ndarray = feature_data.scores
        valid_mask: np.ndarray = feature_data.valid_mask
        n_elements = len(valid_mask)

        # Update consecutive counters based on valid_mask
        for i in range(n_elements):
            if valid_mask[i]:
                # Element is valid (non-NaN)
                self._consecutive_count[i] += 1
                self._gap_count[i] = 0  # Reset gap counter
            else:
                # Element is invalid (NaN)
                if self._visible[i] and self._gap_count[i] < self._config.max_gap_frames:
                    # In grace period - allow brief dropout
                    self._gap_count[i] += 1
                else:
                    # Outside grace period or not yet visible - reset counter
                    self._consecutive_count[i] = 0
                    self._gap_count[i] = 0

        # Update visibility: visible when consecutive count meets minimum
        self._visible = self._consecutive_count >= self._config.min_consecutive_frames

        # Apply the visibility mask
        # For 2D data (points), broadcast mask to match shape (17,) -> (17, 2)
        if values.ndim > 1:
            filtered_mask = self._visible[:, np.newaxis]
        else:
            filtered_mask = self._visible

        filtered_values: np.ndarray = np.where(filtered_mask, values, np.nan)
        # Keep original scores for visible values, set to 0 for hidden values
        filtered_scores: np.ndarray = np.where(self._visible, scores, 0.0).astype(np.float32)

        # Create new feature data with filtered values
        filtered_data = type(feature_data)(values=filtered_values, scores=filtered_scores)

        # Return new pose with filtered feature
        return replace(pose, {self._feature_type: filtered_data})


# Convenience classes for temporal stability filtering
class BBoxTemporalStabilizer(TemporalStabilizer):
    def __init__(self, config: TemporalStabilizerSettings) -> None:
        super().__init__(config, BBox)


class PointTemporalStabilizer(TemporalStabilizer):
    def __init__(self, config: TemporalStabilizerSettings) -> None:
        super().__init__(config, Points2D)


class AngleTemporalStabilizer(TemporalStabilizer):
    def __init__(self, config: TemporalStabilizerSettings) -> None:
        super().__init__(config, Angles)


class AngleVelTemporalStabilizer(TemporalStabilizer):
    def __init__(self, config: TemporalStabilizerSettings) -> None:
        super().__init__(config, AngleVelocity)


class AngleSymTemporalStabilizer(TemporalStabilizer):
    def __init__(self, config: TemporalStabilizerSettings) -> None:
        super().__init__(config, AngleSymmetry)
