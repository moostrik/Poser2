# Standard library imports
from dataclasses import replace

import numpy as np

# Pose imports
from modules.pose.nodes.Nodes import FilterNode, NodeConfigBase
from modules.pose.Frame import Frame, FrameField


class DualConfFilterConfig(NodeConfigBase):
    """Configuration for hysteresis confidence filtering with automatic change notification.

    Hysteresis filtering uses dual thresholds to prevent flickering when scores
    hover near a single threshold. Once a value exceeds threshold_high, it stays
    visible until it drops below threshold_low.
    """

    def __init__(
        self,
        threshold_low: float = 0.3,
        threshold_high: float = 0.5,
        rescale_scores: bool = True
    ) -> None:
        """
        Args:
            threshold_low: Lower threshold - values below this turn off.
                          Range: [0.0, 0.99] (clamped automatically)
            threshold_high: Upper threshold - values above this turn on.
                           Must be >= threshold_low.
                           Range: [0.0, 0.99] (clamped automatically)
            rescale_scores: If True, rescale remaining scores to [0, 1] range.
                          If False, keep original scores for visible values.
        """
        super().__init__()
        self.threshold_low: float = max(0.0, min(0.99, threshold_low))
        self.threshold_high: float = max(0.0, min(0.99, threshold_high))

        # Ensure high >= low
        if self.threshold_high < self.threshold_low:
            self.threshold_high = self.threshold_low

        self.rescale_scores: bool = rescale_scores


class DualConfidenceFilter(FilterNode):
    """Hysteresis-based confidence filter for pose features.

    Implements dual-threshold filtering to prevent flickering:
    - Values turn ON when score >= threshold_high
    - Values turn OFF when score < threshold_low
    - Once on, values stay on until they drop below threshold_low

    This creates a "sticky" behavior that provides stability when scores
    fluctuate around a threshold boundary.

    Args:
        config: Hysteresis filter configuration
        pose_field: FrameField enum specifying which feature to filter

    Example:
        config = HysteresisFilterConfig(threshold_low=0.3, threshold_high=0.5)
        filter = FeatureHysteresisFilter(config, FrameField.points)
    """

    def __init__(self, config: DualConfFilterConfig, pose_field: FrameField):
        self._config = config
        self._pose_field = pose_field
        # State: tracks which elements are currently "on" (visible)
        n_elements = pose_field.get_length()
        self._state_on: np.ndarray = np.zeros(n_elements, dtype=bool)

    @property
    def config(self) -> DualConfFilterConfig:
        return self._config

    def reset(self) -> None:
        """Reset filter state. Called when pose track is lost."""
        n_elements = len(self._state_on)
        self._state_on = np.zeros(n_elements, dtype=bool)

    def process(self, pose: Frame) -> Frame:
        """Filter values using hysteresis thresholds."""
        feature_data = pose.get_feature(self._pose_field)

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
        turn_on = scores >= self._config.threshold_high
        turn_off = scores < self._config.threshold_low

        # Update state: turn on takes priority, then turn off, else maintain
        self._state_on = np.where(turn_on, True, np.where(turn_off, False, self._state_on))

        # Apply the hysteresis mask
        # For 2D data (points), broadcast mask to match shape (17,) -> (17, 2)
        if values.ndim > 1:
            filtered_mask = self._state_on[:, np.newaxis]
        else:
            filtered_mask = self._state_on

        filtered_values: np.ndarray = np.where(filtered_mask, values, np.nan)

        # Handle scores based on rescale_scores config
        if self._config.rescale_scores:
            # Rescale visible scores to [0, 1] range based on high threshold
            # Division by (1 - threshold_high) remaps [threshold_high, 1.0] -> [0, 1.0]
            filtered_scores: np.ndarray = np.where(
                self._state_on,
                (scores - self._config.threshold_low) / (1.0 - self._config.threshold_low),
                0.0
            ).astype(np.float32)
        else:
            # Keep original scores for visible values, set to 0 for hidden values
            filtered_scores: np.ndarray = np.where(self._state_on, scores, 0.0).astype(np.float32)

        # Create new feature data with filtered values
        filtered_data = type(feature_data)(values=filtered_values, scores=filtered_scores)

        # Return new pose with filtered feature
        return replace(pose, **{self._pose_field.name: filtered_data})


# Convenience classes for hysteresis filtering
class BBoxDualConfFilter(DualConfidenceFilter):
    def __init__(self, config: DualConfFilterConfig) -> None:
        super().__init__(config, FrameField.bbox)


class PointDualConfFilter(DualConfidenceFilter):
    def __init__(self, config: DualConfFilterConfig) -> None:
        super().__init__(config, FrameField.points)


class AngleDualConfFilter(DualConfidenceFilter):
    def __init__(self, config: DualConfFilterConfig) -> None:
        super().__init__(config, FrameField.angles)


class AngleVelDualConfFilter(DualConfidenceFilter):
    def __init__(self, config: DualConfFilterConfig) -> None:
        super().__init__(config, FrameField.angle_vel)


class AngleSymDualConfidenceFilter(DualConfidenceFilter):
    def __init__(self, config: DualConfFilterConfig) -> None:
        super().__init__(config, FrameField.angle_sym)