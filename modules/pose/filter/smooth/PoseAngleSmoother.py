# Standard library imports
from dataclasses import replace

# Third-party imports
import numpy as np

# Pose imports
from modules.pose.features.PoseAngles import PoseAngleData, ANGLE_NUM_JOINTS, AngleJoint
from modules.pose.filter.smooth.PoseSmootherBase import PoseSmootherBase, PoseSmootherConfig
from modules.pose.Pose import Pose
from modules.utils.Smoothing import OneEuroFilterAngular


class PoseAngleSmoother(PoseSmootherBase):
    """Smooths pose joint angles using OneEuroFilterAngular."""

    def _create_state(self) -> tuple[list[OneEuroFilterAngular], np.ndarray]:
        """Create angular filters for all joints and validity tracking."""
        filters = [
            OneEuroFilterAngular(
                self._config.frequency,
                self._config.min_cutoff,
                self._config.beta,
                self._config.d_cutoff
            )
            for _ in range(ANGLE_NUM_JOINTS)
        ]
        prev_valid = np.zeros(ANGLE_NUM_JOINTS, dtype=bool)
        return (filters, prev_valid)

    def _smooth(self, pose: Pose, state: tuple[list[OneEuroFilterAngular], np.ndarray]) -> Pose:
        """Smooth joint angles for one pose."""
        filters, prev_valid = state

        smoothed_angles: np.ndarray = pose.angle_data.values.copy()

        for angle_joint in AngleJoint:
            is_valid = pose.angle_data.valid_mask[angle_joint]
            was_valid = prev_valid[angle_joint]

            if is_valid:
                angle = float(pose.angle_data.values[angle_joint])
                angle_filter = filters[angle_joint]

                # Reset if angle reappeared
                if not was_valid and self._config.reset_on_reappear:
                    angle_filter.reset()

                smoothed_angles[angle_joint] = angle_filter(angle)

            prev_valid[angle_joint] = is_valid

        smoothed_angle_data = PoseAngleData(smoothed_angles, pose.angle_data.scores)
        return replace(pose, angle_data=smoothed_angle_data)

    def _on_config_changed(self) -> None:
        """Update filter parameters when config changes."""
        if self._state is not None:
            filters, _ = self._state
            for angle_filter in filters:
                angle_filter.setParameters(
                    self._config.frequency,
                    self._config.min_cutoff,
                    self._config.beta,
                    self._config.d_cutoff
                )