# Standard library imports
from dataclasses import replace

import numpy as np

# Pose imports
from modules.pose.nodes.Nodes import FilterNode
from modules.pose.features import AngleMotion
from modules.pose.features.Angles import AngleLandmark
from modules.pose.features.AngleMotion import ANGLE_MOTION_NORMALISATION
from modules.pose.frame import Frame
from modules.settings import Settings, Field

from modules.utils.HotReloadMethods import HotReloadMethods


class AngleMotionExtractorSettings(Settings):
    """Configuration for motion extraction with tunable thresholds."""
    noise_threshold: Field[float] = Field(0.1)
    max_threshold:   Field[float] = Field(1.0)
    n_top_motions:   Field[int]   = Field(3)


class AngleMotionExtractor(FilterNode):
    """Extracts a single normalized motion value from angular velocities.

    Computes per-joint motion from angle_vel, applies noise floor and
    joint-specific weights, then averages the top N highest motions to produce a single [0, 1] value.
    """

    def __init__(self, config: AngleMotionExtractorSettings | None = None) -> None:
        super().__init__()
        self._config = config if config is not None else AngleMotionExtractorSettings()
        self._normalisation_factors: np.ndarray = np.array(
            [ANGLE_MOTION_NORMALISATION[landmark] for landmark in AngleLandmark],
            dtype=np.float32
        )
        # hot_reload = HotReloadMethods(self.__class__, True, True)

    def process(self, pose: Frame) -> Frame:
        # Get absolute angular velocities
        motions: np.ndarray = np.abs(pose.angle_vel.values)

        # Remove noise: subtract threshold and clip negative values to 0
        motions = np.maximum(motions - self._config.noise_threshold, 0.0)

        # Normalize by joint-specific factors (gives motion semantic meaning)
        motions *= self._normalisation_factors

        # Scale to [0, 1] range based on max_threshold
        motions /= self._config.max_threshold

        # Get top N motions and average them
        valid_motions = motions[~np.isnan(motions)]
        if len(valid_motions) > 0:
            # Sort and get top n values
            n = min(self._config.n_top_motions, len(valid_motions))
            top_n_motions = np.partition(valid_motions, -n)[-n:]
            # Clamp each motion to [0, 1] before averaging
            top_n_motions = np.clip(top_n_motions, 0.0, 1.0)
            avg_motion = float(np.mean(top_n_motions))
        else:
            avg_motion = 0.0

        # Compute average score from valid joints
        valid_mask = ~np.isnan(pose.angle_vel.values)
        avg_score = float(np.mean(pose.angle_vel.scores[valid_mask])) if np.any(valid_mask) else 0.0

        angle_motion: AngleMotion = AngleMotion.from_value(avg_motion, avg_score)
        return replace(pose, angle_motion=angle_motion)

    def reset(self) -> None:
        pass
