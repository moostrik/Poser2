from dataclasses import replace

import numpy as np

from modules.pose.Pose import Pose
from modules.pose.features.PoseAngles import PoseAngleData
from modules.pose.filters.interpolation.PoseAngleInterpolator import PoseAngleInterpolator, TrackletState


class PoseDeltaInterpolator(PoseAngleInterpolator):
    """Interpolates pose angle deltas using circular Hermite interpolation.

    Inherits from PoseAngleInterpolator since delta values are also angular
    and require the same circular wrapping handling. The only difference is
    which data field is being interpolated (delta_data vs angle_data).

    Usage:
        # At input rate (e.g., 30 FPS from camera)
        interpolator.add_poses(input_poses)

        # At output rate (e.g., 60 FPS for display)
        interpolator.update()  # Generates interpolated output

    Note:
        NaN values in delta data are handled gracefully by the interpolator.
        Invalid deltas (zero scores) maintain their NaN values through interpolation.
        Angular wrapping is handled correctly (e.g., interpolating between -π and π
        goes through 0, not through the middle of the circle).
    """

    def _add_sample(self, pose: Pose, tracklet_id: int) -> None:
        """Add input sample to tracklet's interpolator."""

        state: TrackletState = self._tracklets[tracklet_id]

        # Extract delta values from PoseAngleDeltaData (includes NaN for invalid deltas)
        delta_values: np.ndarray = pose.delta_data.values

        # Add sample to interpolator (it handles its own timing)
        state.angle_interpolator.add_sample(delta_values)

    def _interpolate(self, pose: Pose, tracklet_id: int, current_time: float | None) -> Pose:
        """Generate interpolated pose at current time."""
        state: TrackletState = self._tracklets[tracklet_id]

        # Update interpolator to current time
        state.angle_interpolator.update(current_time)

        # Get interpolated values
        interpolated_deltas: np.ndarray = state.angle_interpolator.interpolated_value

        # Create new PoseAngleDeltaData with interpolated values, preserving original scores
        # Note: Scores remain unchanged - interpolation affects only delta values
        interpolated_delta_data = PoseAngleData(
            values=interpolated_deltas,
            scores=pose.delta_data.scores
        )

        return replace(
            pose,
            delta_data=interpolated_delta_data
        )