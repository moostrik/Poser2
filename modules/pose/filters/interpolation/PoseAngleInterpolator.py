from dataclasses import replace

import numpy as np

from modules.pose.Pose import Pose
from modules.pose.features.PoseAngles import PoseAngleData, ANGLE_NUM_JOINTS
from modules.pose.filters.interpolation.PoseInterpolatorBase import PoseInterpolatorBase
from modules.utils.Interpolation import VectorPredictiveAngleHermite


class TrackletState:
    """State container for a single tracklet's angle interpolation filter."""

    def __init__(self, input_rate: float, num_angles: int, alpha_v: float = 0.45) -> None:
        """Initialize interpolation filter for angle values.

        Args:
            input_rate: Expected sampling rate of pose data (Hz)
            num_angles: Number of angles per pose
            alpha_v: Velocity smoothing factor (0.0 to 1.0)
        """
        # Single interpolator for all angle values (handles circular wrapping)
        self.angle_interpolator = VectorPredictiveAngleHermite(
            input_rate=input_rate,
            vector_size=num_angles,
            alpha_v=alpha_v
        )


class PoseAngleInterpolator(PoseInterpolatorBase):
    """Interpolates pose joint angles using circular Hermite interpolation.

    Provides smooth interpolation of joint angles with proper handling of
    angular wrapping (e.g., -π and π are treated as adjacent values).
    Reduces jitter and provides temporal smoothing between input samples.

    Usage:
        # At input rate (e.g., 30 FPS from camera)
        interpolator.add_poses(input_poses)

        # At output rate (e.g., 60 FPS for display)
        interpolator.update()  # Generates interpolated output

    Note:
        NaN values in angle data are handled gracefully by the interpolator.
        Invalid angles (zero scores) maintain their NaN values through interpolation.
        Angular wrapping is handled correctly (e.g., interpolating between -π and π
        goes through 0, not through the middle of the circle).
    """

    def _create_tracklet_state(self) -> TrackletState:
        """Create initial filter state for a new tracklet."""
        return TrackletState(
            input_rate=self._input_rate,
            num_angles=ANGLE_NUM_JOINTS,
            alpha_v=self._alpha_v
        )

    def _add_sample(self, pose: Pose, tracklet_id: int) -> None:
        """Add input sample to tracklet's interpolator."""
        state: TrackletState = self._tracklets[tracklet_id]

        # Extract angle values from PoseAngleData (includes NaN for invalid angles)
        angle_values: np.ndarray = pose.angle_data.values  # Shape: (ANGLE_NUM_JOINTS,)

        # Add sample to interpolator (it handles its own timing)
        state.angle_interpolator.add_sample(angle_values)

    def _interpolate(self, pose: Pose, tracklet_id: int, current_time: float | None) -> Pose:
        """Generate interpolated pose at current time."""
        state: TrackletState = self._tracklets[tracklet_id]

        # Update interpolator to current time
        state.angle_interpolator.update(current_time)

        # Get interpolated values
        interpolated_angles: np.ndarray = state.angle_interpolator.interpolated_value

        # Create new PoseAngleData with interpolated values, preserving original scores
        # Note: Scores remain unchanged - interpolation affects only angle values
        interpolated_angle_data = PoseAngleData(
            values=interpolated_angles,
            scores=pose.angle_data.scores
        )

        return replace(
            pose,
            angle_data=interpolated_angle_data
        )