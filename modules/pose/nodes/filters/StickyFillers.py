"""Pose hold filters that replace NaN values with last valid values.

Provides "hold" behavior for angles, points, and deltas where NaN values
are replaced by the last valid value seen. Useful for maintaining continuity
when pose detection temporarily fails.
"""

# Standard library imports
from dataclasses import replace
from typing import TypeVar

# Third-party imports
import numpy as np

# Pose imports
from modules.pose.features import BaseFeature, PoseFeatureType
from modules.pose.nodes.Nodes import FilterNode, NodeConfigBase
from modules.pose.Pose import Pose, PoseField

TFeature = TypeVar('TFeature', bound=BaseFeature)


class StickyFillerConfig(NodeConfigBase):
    """Configuration for pose hold filter with automatic change notification."""

    def __init__(self, init_to_zero: bool = False, hold_scores: bool = False) -> None:
        """
        Args:
            init_to_zero: If True, initialize with zeros to prevent NaN at start.
                         If False, first NaN values will remain NaN until valid data arrives.
            hold_scores: If True, preserve last valid scores when holding values.
                        If False, set scores to 0.0 for held (NaN) values.
        """
        super().__init__()
        self.init_to_zero: bool = init_to_zero
        self.hold_scores: bool = hold_scores


class FeatureStickyFiller(FilterNode):
    """Generic sticky filler for pose features."""

    def __init__(self, config: StickyFillerConfig, pose_field: PoseField) -> None:
        if not pose_field.is_feature():
            raise ValueError(f"PoseField '{pose_field.value}' is not a feature field")

        self._config: StickyFillerConfig = config
        self._pose_field: PoseField = pose_field
        self._feature_class: PoseFeatureType = pose_field.get_type()
        self._last_valid = self._initialize_last_valid()

    def _initialize_last_valid(self) -> PoseFeatureType:
        """Initialize last valid state based on config."""
        empty_data = self._feature_class.create_dummy()
        if self._config.init_to_zero:
            values = np.zeros_like(empty_data.values)
            scores = np.ones_like(empty_data.scores)
            return type(empty_data)(values=values, scores=scores)
        return empty_data

    @staticmethod
    def _broadcast_mask(mask: np.ndarray, values: np.ndarray) -> np.ndarray:
        """Broadcast 1D mask to match values shape if needed."""
        return mask[:, np.newaxis] if values.ndim > 1 else mask

    @property
    def config(self) -> StickyFillerConfig:
        return self._config

    def process(self, pose: Pose) -> Pose:
        """Replace invalid values with last valid, update state with new valid values."""
        feature_data: PoseFeatureType = pose.get_feature(self._pose_field)
        valid_mask = feature_data.valid_mask
        invalid_mask = ~valid_mask

        # Broadcast masks for value arrays
        invalid_values_mask = self._broadcast_mask(invalid_mask, feature_data.values)
        valid_values_mask = ~invalid_values_mask

        # Create output with held values
        held_values = np.where(invalid_values_mask, self._last_valid.values, feature_data.values)
        score_replacement = self._last_valid.scores if self._config.hold_scores else 0.0
        held_scores = np.where(invalid_mask, score_replacement, feature_data.scores).astype(np.float32)

        # Update internal state (only valid values)
        updated_values = np.where(valid_values_mask, feature_data.values, self._last_valid.values)
        updated_scores = np.where(valid_mask, feature_data.scores, self._last_valid.scores).astype(np.float32)
        self._last_valid = type(feature_data)(values=updated_values, scores=updated_scores)

        # Return updated pose
        held_data = type(feature_data)(values=held_values, scores=held_scores)
        return replace(pose, **{self._pose_field.name: held_data})

    def reset(self) -> None:
        """Reset to initial state."""
        self._last_valid = self._initialize_last_valid()


# Convenience classes
class BBoxStickyFiller(FeatureStickyFiller):
    def __init__(self, config: StickyFillerConfig) -> None:
        super().__init__(config, PoseField.bbox)


class PointStickyFiller(FeatureStickyFiller):
    def __init__(self, config: StickyFillerConfig) -> None:
        super().__init__(config, PoseField.points)


class AngleStickyFiller(FeatureStickyFiller):
    def __init__(self, config: StickyFillerConfig) -> None:
        super().__init__(config, PoseField.angles)


class AngleVelStickyFiller(FeatureStickyFiller):
    def __init__(self, config: StickyFillerConfig) -> None:
        super().__init__(config, PoseField.angle_vel)


class AngleSymStickyFiller(FeatureStickyFiller):
    def __init__(self, config: StickyFillerConfig) -> None:
        super().__init__(config, PoseField.angle_sym)


