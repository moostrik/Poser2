# Standard library imports
from dataclasses import replace

# Third-party imports
import numpy as np

# Pose imports
from modules.pose.features.PosePoints import PosePointData, POSE_NUM_JOINTS, PoseJoint
from modules.pose.filter.smooth.PoseSmootherBase import PoseSmootherBase
from modules.pose.Pose import Pose
from modules.utils.Smoothing import OneEuroFilter


class PosePointSmoother(PoseSmootherBase):
    """Smooths pose keypoint positions using OneEuroFilter."""

    def _create_state(self) -> tuple[list[tuple[OneEuroFilter, OneEuroFilter]], np.ndarray]:
        """Create filters for all joints (x, y per joint) and validity tracking."""
        filters = [
            (
                OneEuroFilter(self._config.frequency, self._config.min_cutoff, self._config.beta, self._config.d_cutoff),
                OneEuroFilter(self._config.frequency, self._config.min_cutoff, self._config.beta, self._config.d_cutoff)
            )
            for _ in range(POSE_NUM_JOINTS)
        ]
        prev_valid = np.zeros(POSE_NUM_JOINTS, dtype=bool)
        return (filters, prev_valid)

    def _smooth(self, pose: Pose, state: tuple[list[tuple[OneEuroFilter, OneEuroFilter]], np.ndarray]) -> Pose:
        """Smooth keypoint positions for one pose."""
        filters, prev_valid = state

        smoothed_values: np.ndarray = pose.point_data.values.copy()

        for joint in PoseJoint:
            is_valid = pose.point_data.valid_mask[joint]
            was_valid = prev_valid[joint]

            if is_valid:
                x, y = pose.point_data.values[joint]
                x_filter, y_filter = filters[joint]

                # Reset if joint reappeared
                if not was_valid and self._config.reset_on_reappear:
                    x_filter.reset()
                    y_filter.reset()

                smoothed_values[joint] = [x_filter(x), y_filter(y)]

            prev_valid[joint] = is_valid

        smoothed_point_data = PosePointData(smoothed_values, pose.point_data.scores)
        return replace(pose, point_data=smoothed_point_data)

    def _on_config_changed(self) -> None:
        """Update filter parameters when config changes."""
        if self._state is not None:
            filters, _ = self._state
            for x_filter, y_filter in filters:
                x_filter.setParameters(
                    self._config.frequency,
                    self._config.min_cutoff,
                    self._config.beta,
                    self._config.d_cutoff
                )
                y_filter.setParameters(
                    self._config.frequency,
                    self._config.min_cutoff,
                    self._config.beta,
                    self._config.d_cutoff
                )