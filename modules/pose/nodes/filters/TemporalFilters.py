"""Temporal stability filters for pose features.

Provides temporal-based filtering where values must be consistently present
for a minimum number of consecutive frames before being considered valid.
This prevents transient detections and flickering.
"""

# Standard library imports
from dataclasses import replace

import numpy as np

# Pose imports
from modules.pose.nodes.Nodes import FilterNode, NodeConfigBase
from modules.pose.Frame import Frame, FrameField


class TemporalStabilizerConfig(NodeConfigBase):
    """Configuration for temporal stability filtering with automatic change notification.

    Temporal stability filtering requires values to be consistently valid (non-NaN)
    for multiple consecutive frames before they become visible. This prevents
    momentary detections and flickering.

    Note: This filter works on already-filtered data. Apply confidence/hysteresis
    filtering BEFORE this filter in the pipeline.
    """

    def __init__(
        self,
        min_consecutive_frames: int = 3,
        max_gap_frames: int = 0
    ) -> None:
        """
        Args:
            min_consecutive_frames: Number of consecutive frames a value must be valid
                                   before it becomes visible.
                                   Range: [1, 30] (clamped automatically)
            max_gap_frames: Maximum number of frames a value can be invalid
                           before the consecutive counter resets. Allows brief dropouts.
                           Range: [0, 10] (clamped automatically)
        """
        super().__init__()
        self.min_consecutive_frames: int = max(1, min(30, min_consecutive_frames))
        self.max_gap_frames: int = max(0, min(10, max_gap_frames))


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
        pose_field: FrameField enum specifying which feature to filter

    Example:
        config = TemporalStabilityConfig(
            min_consecutive_frames=3,
            max_gap_frames=1
        )
        filter = TemporalStabilityFilter(config, FrameField.points)
    """

    def __init__(self, config: TemporalStabilizerConfig, pose_field: FrameField):
        self._config = config
        self._pose_field = pose_field
        # Get number of elements from pose field
        n_elements = pose_field.get_length()

        # State: tracks consecutive frame count for each element
        # Positive values = consecutive frames valid, 0 = not yet stable
        self._consecutive_count: np.ndarray = np.zeros(n_elements, dtype=np.int32)
        # State: tracks current gap count (frames invalid while in grace period)
        self._gap_count: np.ndarray = np.zeros(n_elements, dtype=np.int32)
        # State: tracks which elements are currently visible
        self._visible: np.ndarray = np.zeros(n_elements, dtype=bool)

    @property
    def config(self) -> TemporalStabilizerConfig:
        return self._config

    def reset(self) -> None:
        """Reset filter state. Called when pose track is lost."""
        n_elements = len(self._consecutive_count)
        self._consecutive_count = np.zeros(n_elements, dtype=np.int32)
        self._gap_count = np.zeros(n_elements, dtype=np.int32)
        self._visible = np.zeros(n_elements, dtype=bool)

    def process(self, pose: Frame) -> Frame:
        """Filter values requiring temporal stability."""
        feature_data = pose.get_feature(self._pose_field)

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
        return replace(pose, **{self._pose_field.name: filtered_data})


# Convenience classes for temporal stability filtering
class BBoxTemporalStabilizer(TemporalStabilizer):
    def __init__(self, config: TemporalStabilizerConfig) -> None:
        super().__init__(config, FrameField.bbox)


class PointTemporalStabilizer(TemporalStabilizer):
    def __init__(self, config: TemporalStabilizerConfig) -> None:
        super().__init__(config, FrameField.points)


class AngleTemporalStabilizer(TemporalStabilizer):
    def __init__(self, config: TemporalStabilizerConfig) -> None:
        super().__init__(config, FrameField.angles)


class AngleVelTemporalStabilizer(TemporalStabilizer):
    def __init__(self, config: TemporalStabilizerConfig) -> None:
        super().__init__(config, FrameField.angle_vel)


class AngleSymTemporalStabilizer(TemporalStabilizer):
    def __init__(self, config: TemporalStabilizerConfig) -> None:
        super().__init__(config, FrameField.angle_sym)
