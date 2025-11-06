# Standard library imports
from dataclasses import replace

# Third-party imports
import numpy as np

# Pose imports
from modules.pose.features.PoseAngles import PoseAngleData, AngleJoint, ANGLE_NUM_JOINTS
from modules.pose.filter.smooth.PoseAngleSmoother import PoseAngleSmoother
from modules.pose.Pose import Pose
from modules.utils.Smoothing import OneEuroFilterAngular


class PoseAngleDeltaSmoother(PoseAngleSmoother):
    """Smooths pose angle deltas using OneEuroFilterAngular.

    Reuses all logic from PoseAngleSmoother but operates on delta_data instead of angle_data.
    """

    def _smooth(self, pose: Pose, state: tuple[list[OneEuroFilterAngular], np.ndarray]) -> Pose:
        """Smooth angle deltas for one pose."""
        filters, prev_valid = state

        # Smooth angle deltas
        smoothed_deltas: np.ndarray = pose.delta_data.values.copy()

        for angle_joint in AngleJoint:
            is_valid = pose.delta_data.valid_mask[angle_joint]
            was_valid = prev_valid[angle_joint]

            if is_valid:
                delta = float(pose.delta_data.values[angle_joint])
                angle_filter = filters[angle_joint]

                # Reset if delta reappeared
                if not was_valid and self._config.reset_on_reappear:
                    angle_filter.reset()

                smoothed_deltas[angle_joint] = angle_filter(delta)

            prev_valid[angle_joint] = is_valid

        smoothed_delta_data = PoseAngleData(smoothed_deltas, pose.delta_data.scores)
        return replace(pose, delta_data=smoothed_delta_data)