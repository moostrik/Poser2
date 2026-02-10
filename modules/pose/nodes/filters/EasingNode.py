"""Easing filter node for applying pytweening functions to normalized values."""

# Standard library imports
from dataclasses import replace
from typing import Callable

# Third-party imports
import pytweening  # type: ignore

# Pose imports
from modules.pose.nodes.Nodes import FilterNode, NodeConfigBase
from modules.pose.features.base import NormalizedSingleValue
from modules.pose.Frame import Frame, FrameField


# Available easing functions from pytweening
EASING_FUNCTIONS: dict[str, Callable[[float], float]] = {
    'linear': pytweening.linear,
    'easeInSine': pytweening.easeInSine,
    'easeOutSine': pytweening.easeOutSine,
    'easeInOutSine': pytweening.easeInOutSine,
    'easeInQuad': pytweening.easeInQuad,
    'easeOutQuad': pytweening.easeOutQuad,
    'easeInOutQuad': pytweening.easeInOutQuad,
    'easeInCubic': pytweening.easeInCubic,
    'easeOutCubic': pytweening.easeOutCubic,
    'easeInOutCubic': pytweening.easeInOutCubic,
    'easeInQuart': pytweening.easeInQuart,
    'easeOutQuart': pytweening.easeOutQuart,
    'easeInOutQuart': pytweening.easeInOutQuart,
    'easeInExpo': pytweening.easeInExpo,
    'easeOutExpo': pytweening.easeOutExpo,
    'easeInOutExpo': pytweening.easeInOutExpo,
    'easeInCirc': pytweening.easeInCirc,
    'easeOutCirc': pytweening.easeOutCirc,
    'easeInOutCirc': pytweening.easeInOutCirc,
}


class EasingConfig(NodeConfigBase):
    """Configuration for easing with selectable function."""

    def __init__(self, easing_name: str = 'easeInOutSine') -> None:
        super().__init__()
        self.easing_name: str = easing_name

    @property
    def easing_function(self) -> Callable[[float], float]:
        """Get the current easing function."""
        return EASING_FUNCTIONS.get(self.easing_name, pytweening.linear)


class EasingNode(FilterNode):
    """Applies an easing function to a NormalizedSingleValue field.

    Works with any Frame field that inherits from NormalizedSingleValue
    (e.g., AngleMotion, LeaderScore).
    """

    def __init__(self, config: EasingConfig, pose_field: FrameField) -> None:
        self._config: EasingConfig = config
        self._pose_field: FrameField = pose_field

    @property
    def config(self) -> EasingConfig:
        return self._config

    def process(self, pose: Frame) -> Frame:
        feature: NormalizedSingleValue = pose.get_feature(self._pose_field)

        # Get original value and apply easing
        original_value = feature.value
        if original_value != original_value:  # NaN check
            return pose

        # Clamp to [0, 1] before applying easing
        clamped = max(0.0, min(1.0, original_value))
        eased_value = self._config.easing_function(clamped)

        # Create new feature with eased value
        feature_type = type(feature)
        if hasattr(feature_type, 'from_value'):
            new_feature = feature_type.from_value(eased_value, feature.score)
        else:
            # Fallback for features without from_value
            import numpy as np
            new_feature = feature_type(
                values=np.array([eased_value], dtype=np.float32),
                scores=feature.scores
            )

        return replace(pose, **{self._pose_field.name: new_feature})


class AngleMotionEasingNode(EasingNode):
    """Convenience class for easing angle_motion."""

    def __init__(self, config: EasingConfig) -> None:
        super().__init__(config, FrameField.angle_motion)
