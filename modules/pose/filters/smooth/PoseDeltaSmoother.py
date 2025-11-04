# Standard library imports
from dataclasses import replace

# Third-party imports
import numpy as np

# Pose imports
from .PoseAngleSmoother import PoseAngleSmoother
from ...features.PoseAngles import PoseAngleData, AngleJoint
from ...Pose import Pose


class PoseAngleDeltaSmoother(PoseAngleSmoother):
    """Smooths pose angle deltas using OneEuroFilterAngular.

    Reuses all logic from PoseSmoothAngleFilter but operates on angle_delta_data
    instead of angle_data.
    """

    def _smooth(self, pose: Pose, tracklet_id: int) -> Pose:
        """Smooth angle deltas for one pose."""
        filters, prev_valid = self._tracklets[tracklet_id]

        # Smooth angle deltas
        smoothed_deltas: np.ndarray = pose.delta_data.values.copy()
        smoothed_deltas.flags.writeable = True

        for angle_joint in AngleJoint:
            is_valid = pose.delta_data.valid_mask[angle_joint]
            was_valid = prev_valid[angle_joint]

            if is_valid:
                delta = float(pose.delta_data.values[angle_joint])
                angle_filter = filters[angle_joint]

                # Reset if delta reappeared
                if not was_valid and self.settings.reset_on_reappear:
                    angle_filter.reset()

                smoothed_deltas[angle_joint] = angle_filter(delta)

            prev_valid[angle_joint] = is_valid

        smoothed_delta_data = PoseAngleData(smoothed_deltas, pose.delta_data.scores)
        return replace(pose, delta_data=smoothed_delta_data)