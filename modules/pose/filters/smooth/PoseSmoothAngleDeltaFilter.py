from dataclasses import replace

import numpy as np

from modules.pose.Pose import Pose
from modules.pose.features.PoseAngles import PoseAngleData, AngleJoint
from modules.pose.filters.smooth.PoseSmoothAngleFilter import PoseSmoothAngleFilter


class PoseSmoothAngleDeltaFilter(PoseSmoothAngleFilter):
    """Smooths pose angle deltas using OneEuroFilterAngular.

    Reuses all logic from PoseSmoothAngleFilter but operates on angle_delta_data
    instead of angle_data.
    """

    def _smooth_pose(self, pose: Pose, tracklet_id: int) -> Pose:
        """Smooth angle deltas for one pose."""
        filters, prev_valid = self._tracklets[tracklet_id]
        timestamp: float = pose.time_stamp.timestamp()

        # Smooth angle deltas
        smoothed_deltas: np.ndarray = pose.angle_delta_data.values.copy()
        smoothed_deltas.flags.writeable = True

        for angle_joint in AngleJoint:
            is_valid = pose.angle_delta_data.valid_mask[angle_joint]
            was_valid = prev_valid[angle_joint]

            if is_valid:
                delta = float(pose.angle_delta_data.values[angle_joint])
                angle_filter = filters[angle_joint]

                # Reset if delta reappeared
                if not was_valid and self.settings.reset_on_reappear:
                    angle_filter.reset()

                smoothed_deltas[angle_joint] = angle_filter(delta, timestamp)

            prev_valid[angle_joint] = is_valid

        smoothed_angle_delta_data = PoseAngleData(smoothed_deltas, pose.angle_delta_data.scores)
        return replace(pose, angle_delta_data=smoothed_angle_delta_data)