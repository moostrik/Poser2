from dataclasses import replace

import numpy as np

from modules.pose.Pose import Pose
from modules.pose.features.PoseAngles import PoseAngleData, ANGLE_NUM_JOINTS, AngleJoint
from modules.pose.filters.smooth.PoseSmoothFilterBase import PoseSmoothFilterBase
from modules.utils.Smoothing import OneEuroFilterAngular
from modules.Settings import Settings


class PoseSmoothAngleFilter(PoseSmoothFilterBase):
    """Smooths pose joint angles using OneEuroFilterAngular."""

    def _create_tracklet_state(self) -> tuple[list[OneEuroFilterAngular], np.ndarray]:
        """Create angular filters for all joints and validity tracking."""
        filters = [
            OneEuroFilterAngular(self.settings.frequency, self.settings.min_cutoff, self.settings.beta, self.settings.d_cutoff)
            for _ in range(ANGLE_NUM_JOINTS)
        ]
        prev_valid = np.zeros(ANGLE_NUM_JOINTS, dtype=bool)
        return (filters, prev_valid)

    def _smooth_pose(self, pose: Pose, tracklet_id: int) -> Pose:
        """Smooth joint angles for one pose."""
        filters, prev_valid = self._tracklets[tracklet_id]
        timestamp: float = pose.time_stamp.timestamp()

        # Smooth angles
        smoothed_angles: np.ndarray = pose.angle_data.values.copy()
        smoothed_angles.flags.writeable = True

        for angle_joint in AngleJoint:
            is_valid = pose.angle_data.valid_mask[angle_joint]
            was_valid = prev_valid[angle_joint]

            if is_valid:
                angle = float(pose.angle_data.values[angle_joint])
                angle_filter = filters[angle_joint]

                # Reset if angle reappeared
                if not was_valid and self.settings.reset_on_reappear:
                    angle_filter.reset()

                smoothed_angles[angle_joint] = angle_filter(angle, timestamp)

            prev_valid[angle_joint] = is_valid

        smoothed_angle_data = PoseAngleData(smoothed_angles, pose.angle_data.scores)
        return replace(pose, angle_data=smoothed_angle_data)

    def _update_tracklet_filters(self, tracklet_state: tuple[list[OneEuroFilterAngular], np.ndarray]) -> None:
        """Update filter parameters for all angle filters."""
        filters, _ = tracklet_state
        for angle_filter in filters:
            angle_filter.setParameters(self.settings.frequency, self.settings.min_cutoff, self.settings.beta, self.settings.d_cutoff)