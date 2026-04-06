"""Pose hold filters that replace NaN values with last valid values.

Provides "hold" behavior for angles, points, and deltas where NaN values
are replaced by the last valid value seen. Useful for maintaining continuity
when pose detection temporarily fails.
"""

# Third-party imports
import numpy as np

# Pose imports
from modules.pose.features import Angles, AngleVelocity, AngleSymmetry, BBox, Points2D, Similarity
from modules.pose.features.base import BaseFeature
from modules.pose.nodes.Nodes import FilterNode
from modules.pose.frame import Frame, replace
from modules.settings import BaseSettings, Field


class StickyFillerSettings(BaseSettings):
    """Configuration for pose hold filter."""
    init_to_zero: Field[bool] = Field(False)
    hold_scores:  Field[bool] = Field(False)


class FeatureStickyFiller(FilterNode):
    """Generic sticky filler for pose features."""

    def __init__(self, config: StickyFillerSettings, feature_type: type[BaseFeature]) -> None:

        self._config: StickyFillerSettings = config
        self._feature_type: type[BaseFeature] = feature_type
        self._last_valid = self._initialize_last_valid()

    def _initialize_last_valid(self) -> BaseFeature:
        """Initialize last valid state based on config."""
        empty_data = self._feature_type.create_dummy()
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
    def config(self) -> StickyFillerSettings:
        return self._config

    def process(self, pose: Frame) -> Frame:
        """Replace invalid values with last valid, update state with new valid values."""
        feature_data = pose[self._feature_type]
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
        return replace(pose, {self._feature_type: held_data})

    def reset(self) -> None:
        """Reset to initial state."""
        self._last_valid = self._initialize_last_valid()


# Convenience classes
class BBoxStickyFiller(FeatureStickyFiller):
    def __init__(self, config: StickyFillerSettings) -> None:
        super().__init__(config, BBox)


class PointStickyFiller(FeatureStickyFiller):
    def __init__(self, config: StickyFillerSettings) -> None:
        super().__init__(config, Points2D)


class AngleStickyFiller(FeatureStickyFiller):
    def __init__(self, config: StickyFillerSettings) -> None:
        super().__init__(config, Angles)


class AngleVelStickyFiller(FeatureStickyFiller):
    def __init__(self, config: StickyFillerSettings) -> None:
        super().__init__(config, AngleVelocity)


class AngleSymStickyFiller(FeatureStickyFiller):
    def __init__(self, config: StickyFillerSettings) -> None:
        super().__init__(config, AngleSymmetry)

class SimilarityStickyFiller(FeatureStickyFiller):
    def __init__(self, config: StickyFillerSettings) -> None:
        super().__init__(config, Similarity)
