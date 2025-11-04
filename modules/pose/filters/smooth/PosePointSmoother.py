from dataclasses import replace

import numpy as np

from modules.pose.Pose import Pose
from modules.pose.features.PosePoints import PosePointData, POSE_NUM_JOINTS, PoseJoint
from modules.pose.filters.smooth.PoseSmootherBase import PoseSmootherBase
from modules.utils.Smoothing import OneEuroFilter


class PosePointSmoother(PoseSmootherBase):
    """Smooths pose keypoint positions using OneEuroFilter."""

    def _create_tracklet_state(self) -> tuple[list[tuple[OneEuroFilter, OneEuroFilter]], np.ndarray]:
        """Create filters for all joints (x, y per joint) and validity tracking."""
        filters = [
            (OneEuroFilter(self.settings.frequency, self.settings.min_cutoff, self.settings.beta, self.settings.d_cutoff),
             OneEuroFilter(self.settings.frequency, self.settings.min_cutoff, self.settings.beta, self.settings.d_cutoff))
            for _ in range(POSE_NUM_JOINTS)
        ]
        prev_valid = np.zeros(POSE_NUM_JOINTS, dtype=bool)
        return (filters, prev_valid)

    def _smooth_pose(self, pose: Pose, tracklet_id: int) -> Pose:
        """Smooth keypoint positions for one pose."""
        filters, prev_valid = self._tracklets[tracklet_id]

        # Smooth points
        smoothed_values: np.ndarray = pose.point_data.values.copy()
        smoothed_values.flags.writeable = True

        for joint in PoseJoint:
            is_valid = pose.point_data.valid_mask[joint]
            was_valid = prev_valid[joint]

            if is_valid:
                x, y = pose.point_data.values[joint]
                x_filter, y_filter = filters[joint]

                # Reset if joint reappeared
                if not was_valid and self.settings.reset_on_reappear:
                    x_filter.reset()
                    y_filter.reset()

                smoothed_values[joint] = [x_filter(x), y_filter(y)]

            prev_valid[joint] = is_valid

        smoothed_point_data = PosePointData(smoothed_values, pose.point_data.scores)
        return replace(pose, point_data=smoothed_point_data)

    def _update_tracklet_filters(self, tracklet_state: tuple[list[tuple[OneEuroFilter, OneEuroFilter]], np.ndarray]) -> None:
        """Update filter parameters for all joint filters."""
        filters, _ = tracklet_state
        for x_filter, y_filter in filters:
            x_filter.setParameters(self.settings.frequency, self.settings.min_cutoff, self.settings.beta, self.settings.d_cutoff)
            y_filter.setParameters(self.settings.frequency, self.settings.min_cutoff, self.settings.beta, self.settings.d_cutoff)