from dataclasses import replace

import numpy as np

from modules.pose.Pose import Pose
from modules.pose.filters.interpolation.PoseInterpolatorBase import PoseInterpolatorBase
from modules.utils.Interpolation import VectorPredictiveHermite
from modules.utils.PointsAndRects import Rect


class TrackletState:
    """State container for a single tracklet's bounding box interpolation filter."""

    def __init__(self, input_rate: float, alpha_v: float = 0.45) -> None:
        """Initialize interpolation filter for bounding box parameters.

        Args:
            input_rate: Expected sampling rate of pose data (Hz)
            alpha_v: Velocity smoothing factor (0.0 to 1.0)
        """
        # Single interpolator for all 4 bounding box parameters: [x, y, width, height]
        self.bbox_interpolator = VectorPredictiveHermite(
            input_rate=input_rate,
            vector_size=4,
            alpha_v=alpha_v
        )


class PoseBBoxInterpolator(PoseInterpolatorBase):
    """Interpolates pose bounding box coordinates using cubic Hermite interpolation.

    Provides smooth interpolation of bounding box position (x, y) and size (width, height),
    reducing jitter and providing temporal smoothing between input samples.

    Usage:
        # At input rate (e.g., 30 FPS from camera)
        interpolator.add_poses(input_poses)

        # At output rate (e.g., 60 FPS for display)
        interpolator.update()  # Generates interpolated output

    Note:
        Bounding box coordinates are in normalized space [0, 1] but can extend
        beyond these bounds for crops that extend past image edges.
        All four parameters (x, y, width, height) are interpolated simultaneously.
    """

    def _create_tracklet_state(self) -> TrackletState:
        """Create initial filter state for a new tracklet."""
        return TrackletState(
            input_rate=self._input_rate,
            alpha_v=self._alpha_v
        )

    def _add_sample(self, pose: Pose, tracklet_id: int) -> None:
        """Add input sample to tracklet's interpolator."""
        state: TrackletState = self._tracklets[tracklet_id]

        # Extract bounding box parameters as array: [x, y, width, height]
        bbox: Rect = pose.bounding_box
        bbox_array: np.ndarray = np.array([bbox.x, bbox.y, bbox.width, bbox.height])

        # Add sample to interpolator (it handles its own timing)
        state.bbox_interpolator.add_sample(bbox_array)

    def _interpolate(self, pose: Pose, tracklet_id: int, current_time: float | None) -> Pose:
        """Generate interpolated pose at current time."""
        state: TrackletState = self._tracklets[tracklet_id]

        # Update interpolator to current time
        state.bbox_interpolator.update(current_time)

        # Get interpolated values: [x, y, width, height]
        interpolated_bbox: np.ndarray = state.bbox_interpolator.interpolated_value

        # Create new Rect with interpolated values
        interpolated_rect = Rect(
            x=interpolated_bbox[0],
            y=interpolated_bbox[1],
            width=interpolated_bbox[2],
            height=interpolated_bbox[3]
        )

        return replace(
            pose,
            bounding_box=interpolated_rect
        )