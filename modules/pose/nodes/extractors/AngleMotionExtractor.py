# Standard library imports
from dataclasses import replace

import numpy as np

# Pose imports
from modules.pose.nodes.Nodes import FilterNode
from modules.pose.features import AngleVelocity, AngleMotion
from modules.pose.features.Angles import AngleLandmark
from modules.pose.features.AngleMotion import AngleMotion, ANGLE_MOTION_NORMALISATION
from modules.pose.Frame import Frame


from modules.utils.HotReloadMethods import HotReloadMethods


class AngleMotionExtractor(FilterNode):
    """Computes frame-to-frame changes (deltas) for a single pose.

    Calculates:
    - Angle displacement: Angular change (with proper wrapping) since last frame

    Handles occlusion: Sets deltas to NaN when joints reappear after being invalid.
    """

    def __init__(self) -> None:
        super().__init__()
        self._prev_pose: Frame | None = None
        self.normalisation_factors: np.ndarray = np.array(
            [ANGLE_MOTION_NORMALISATION[landmark] for landmark in AngleLandmark],
            dtype=np.float32
        )

        hot_reload = HotReloadMethods(self.__class__, True, True)

    def process(self, pose: Frame) -> Frame:
        # Compute deltas (or empty if no previous pose)

        self.normalisation_factors[0] = 1.0
        self.normalisation_factors[1] = 1.0
        self.normalisation_factors[2] = 1.0
        self.normalisation_factors[3] = 1.0
        self.normalisation_factors[4] = 2.0
        self.normalisation_factors[5] = 2.0
        self.normalisation_factors[6] = 2.0
        self.normalisation_factors[7] = 2.0
        self.normalisation_factors[8] = 3.0

        self.noise_threshold: float = 0.3  # Noise floor - ignore motion below this
        self.max_threshold: float = np.pi  # Upper limit for normalization

        motion: np.ndarray = np.abs(pose.angle_vel.values)

        # Remove noise: subtract threshold and clip negative values to 0
        motion = np.maximum(motion - self.noise_threshold, 0.0)

        # Normalize by joint-specific factors (gives motion semantic meaning)
        motion *= self.normalisation_factors

        # Scale to [0, 1] range based on max_threshold
        motion /= self.max_threshold
        motion = np.clip(motion, 0.0, 1.0)


        if np.isnan(motion).all():
            max_motion = np.nan
        else:
            max_motion = np.nanmax(motion)
        max_motion_array = np.full_like(motion, max_motion)

        # print(max_motion)

        angle_motion: AngleMotion = AngleMotion(values=motion, scores=pose.angle_vel.scores)
        enriched_pose: Frame = replace(
            pose,
            angle_motion=angle_motion
        )

        return enriched_pose

    def reset(self) -> None:
        self._prev_pose = None
