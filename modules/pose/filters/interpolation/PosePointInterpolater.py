from dataclasses import replace

import numpy as np

from modules.pose.Pose import Pose
from modules.pose.features.PosePoints import PosePointData, POSE_NUM_JOINTS
from modules.pose.filters.interpolation.PoseInterpolatorBase import PoseInterpolaterBase
from modules.utils.Interpolation import VectorPredictiveHermite


class TrackletState:
    """State container for a single tracklet's interpolation filters."""

    def __init__(self, input_rate: float, num_keypoints: int, alpha_v: float = 0.45) -> None:
        """Initialize interpolation filters for x and y coordinates.

        Args:
            input_rate: Expected sampling rate of pose data (Hz)
            num_keypoints: Number of keypoints per pose
            alpha_v: Velocity smoothing factor (0.0 to 1.0)
        """
        # Separate interpolators for x and y coordinates
        self.x_interpolator = VectorPredictiveHermite(
            input_rate=input_rate,
            vector_size=num_keypoints,
            alpha_v=alpha_v
        )
        self.y_interpolator = VectorPredictiveHermite(
            input_rate=input_rate,
            vector_size=num_keypoints,
            alpha_v=alpha_v
        )


class PosePointInterpolater(PoseInterpolaterBase):
    """Interpolates pose keypoint coordinates using cubic Hermite interpolation.

    Provides smooth interpolation of x,y coordinates for all pose keypoints,
    reducing jitter and providing temporal smoothing between input samples.

    Usage:
        # At input rate (e.g., 30 FPS from camera)
        interpolater.add_poses(input_poses)

        # At output rate (e.g., 60 FPS for display)
        interpolater.update()  # Generates interpolated output

    Note:
        NaN values in pose data are handled gracefully by the interpolators.
        Invalid keypoints (zero scores) maintain their NaN values through interpolation.
    """

    def _create_tracklet_state(self) -> TrackletState:
        """Create initial filter state for a new tracklet.

        Returns:
            TrackletState containing x and y interpolators
        """
        return TrackletState(
            input_rate=self.input_rate,
            num_keypoints=POSE_NUM_JOINTS,
            alpha_v=self.alpha_v
        )

    def _add_sample(self, pose: Pose, tracklet_id: int) -> None:
        """Add input sample to tracklet's interpolators.

        Args:
            pose: Input pose to sample
            tracklet_id: ID of the tracklet (for accessing filter state)
        """
        state: TrackletState = self._tracklets[tracklet_id]

        # Extract coordinates from PosePointData (includes NaN for invalid joints)
        x_coords: np.ndarray = pose.point_data.values[:, 0]  # Shape: (POSE_NUM_JOINTS,)
        y_coords: np.ndarray = pose.point_data.values[:, 1]  # Shape: (POSE_NUM_JOINTS,)

        # Add samples to interpolators (they handle their own timing)
        state.x_interpolator.add_sample(x_coords)
        state.y_interpolator.add_sample(y_coords)

    def _interpolate(self, pose: Pose, tracklet_id: int, current_time: float | None) -> Pose:
        """Generate interpolated pose at current time.

        Args:
            pose: Last input pose (used as template for reconstruction)
            tracklet_id: ID of the tracklet (for accessing filter state)
            current_time: Optional explicit time for interpolation

        Returns:
            Pose with interpolated keypoint coordinates

        Note:
            Invalid keypoints (NaN values, zero scores) are passed through the
            interpolators, which handle NaN gracefully element-wise.
        """
        state: TrackletState = self._tracklets[tracklet_id]

        # Update interpolators to current time
        state.x_interpolator.update(current_time)
        state.y_interpolator.update(current_time)

        # Get interpolated values
        interpolated_x: np.ndarray = state.x_interpolator.interpolated_value
        interpolated_y: np.ndarray = state.y_interpolator.interpolated_value

        # Reconstruct values array (POSE_NUM_JOINTS, 2)
        interpolated_values: np.ndarray = np.stack([interpolated_x, interpolated_y], axis=1)

        # Create new PosePointData with interpolated values, preserving original scores
        # Note: Scores remain unchanged - interpolation affects only positions
        interpolated_points = PosePointData(
            values=interpolated_values,
            scores=pose.point_data.scores
        )

        return replace(
            pose,
            point_data=interpolated_points
        )