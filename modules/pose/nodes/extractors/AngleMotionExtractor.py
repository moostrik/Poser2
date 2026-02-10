# Standard library imports
from dataclasses import replace

import numpy as np

# Pose imports
from modules.pose.nodes.Nodes import FilterNode, NodeConfigBase
from modules.pose.features import AngleMotion
from modules.pose.features.Angles import AngleLandmark
from modules.pose.features.AngleMotion import ANGLE_MOTION_NORMALISATION
from modules.pose.Frame import Frame

from modules.utils.HotReloadMethods import HotReloadMethods


class AngleMotionExtractorConfig(NodeConfigBase):
    """Configuration for motion extraction with tunable thresholds."""

    def __init__(
        self,
        noise_threshold: float = 0.075,
        max_threshold: float = 0.5,
    ) -> None:
        super().__init__()
        self.noise_threshold: float = noise_threshold  # Noise floor - ignore motion below this
        self.max_threshold: float = max_threshold      # Upper limit for normalization


class AngleMotionExtractor(FilterNode):
    """Extracts a single normalized motion value from angular velocities.

    Computes per-joint motion from angle_vel, applies noise floor and
    joint-specific weights, then takes MAX to produce a single [0, 1] value.
    """

    def __init__(self, config: AngleMotionExtractorConfig | None = None) -> None:
        super().__init__()
        self._config = config if config is not None else AngleMotionExtractorConfig()
        self._normalisation_factors: np.ndarray = np.array(
            [ANGLE_MOTION_NORMALISATION[landmark] for landmark in AngleLandmark],
            dtype=np.float32
        )
        hot_reload = HotReloadMethods(self.__class__, True, True)

    def process(self, pose: Frame) -> Frame:
        # Get absolute angular velocities
        motion: np.ndarray = np.abs(pose.angle_vel.values)

        # Remove noise: subtract threshold and clip negative values to 0
        motion = np.maximum(motion - self._config.noise_threshold, 0.0)

        # Normalize by joint-specific factors (gives motion semantic meaning)
        motion *= self._normalisation_factors

        # Scale to [0, 1] range based on max_threshold
        motion /= self._config.max_threshold

        # Take MAX across all joints - this is our single motion value
        max_motion: float = float(np.nanmax(motion)) if not np.all(np.isnan(motion)) else 0.0
        max_motion = min(1.0, max(0.0, max_motion))

        # Compute average score from valid joints
        valid_mask = ~np.isnan(pose.angle_vel.values)
        avg_score = float(np.mean(pose.angle_vel.scores[valid_mask])) if np.any(valid_mask) else 0.0

        angle_motion: AngleMotion = AngleMotion.from_value(max_motion, avg_score)
        return replace(pose, angle_motion=angle_motion)

    def reset(self) -> None:
        pass
