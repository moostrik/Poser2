from typing import Optional
from threading import Lock

from modules.pose.Pose import Pose, PoseDict

from ..PoseFilterBase import PoseFilterBase
from .PosePointInterpolator import PosePointInterpolator
from .PoseAngleInterpolator import PoseAngleInterpolator
from .PoseDeltaInterpolator import PoseDeltaInterpolator
from .PoseBBoxInterpolator import PoseBBoxInterpolator

from modules.Settings import Settings


class PoseInterpolator(PoseFilterBase):
    """Combines multiple pose feature interpolators for complete pose interpolation."""

    def __init__(self, settings: Settings) -> None:
        """Initialize all feature interpolators."""
        super().__init__()
        # Initialize feature interpolators
        self._point_interpolator = PosePointInterpolator(settings)
        self._angle_interpolator = PoseAngleInterpolator(settings)
        self._delta_interpolator = PoseDeltaInterpolator(settings)
        self._bbox_interpolator = PoseBBoxInterpolator(settings)

        # Register callbacks
        self._point_interpolator.add_callback(self._angle_interpolator.add_poses)
        self._angle_interpolator.add_callback(self._delta_interpolator.add_poses)
        self._delta_interpolator.add_callback(self._bbox_interpolator.add_poses)
        self._bbox_interpolator.add_callback(self._notify_callbacks)

        # Thread-safe storage for interpolated poses
        self._result_lock = Lock()
        self._interpolated_poses: PoseDict = {}

    def add_poses(self, poses: PoseDict) -> None:
        """Add input samples to all interpolators (called at input rate)."""
        if not poses:
            return  # No poses to process
        self._point_interpolator.add_poses(poses)
        self._angle_interpolator.add_poses(poses)
        self._delta_interpolator.add_poses(poses)
        self._bbox_interpolator.add_poses(poses)

    def update(self, current_time: float | None = None) -> None:
        """Update output for all poses (called at output rate)."""
        # calling the first interpolator, propagates to others via callbacks
        self._point_interpolator.update(current_time)

    def get_pose(self, pose_id: int) -> Optional[Pose]:
        """Get interpolated pose by ID."""
        with self._result_lock:
            return self._interpolated_poses.get(pose_id)

    def get_poses(self) -> PoseDict:
        """Get all currently interpolated poses."""
        with self._result_lock:
            return self._interpolated_poses

    def has_pose(self, pose_id: int) -> bool:
        """Check if a pose exists in current interpolation."""
        with self._result_lock:
            return pose_id in self._interpolated_poses

    def get_pose_ids(self) -> list[int]:
        """Get list of all available pose IDs."""
        with self._result_lock:
            return list(self._interpolated_poses.keys())

    def _notify_callbacks(self, poses: PoseDict) -> None:
        """Notify all registered callbacks and collect interpolated poses."""
        with self._result_lock:
            self._interpolated_poses.update(poses)
        super()._notify_callbacks(poses)
