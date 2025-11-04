from typing import Optional
from modules.pose.Pose import Pose, PoseDict
from modules.pose.filters.interpolation.PosePointInterpolater import PosePointInterpolater
from modules.Settings import Settings


class PoseInterpolater:
    """Combines multiple pose feature interpolators for complete pose interpolation.

    Orchestrates interpolation of different pose features (points, angles, etc.)
    with separated input and output rates:
    - add_poses(): Feed input samples at camera rate
    - update(): Generate interpolated output at display rate
    - get(): Retrieve interpolated poses by ID

    Usage:
        # Initialize with settings
        interpolater = PoseInterpolater(settings)

        # At input rate (e.g., 30 FPS)
        interpolater.add_poses(input_poses)

        # At output rate (e.g., 60 FPS)
        interpolater.update()
        interpolated_pose = interpolater.get(pose_id)
        all_poses = interpolater.get_all()
    """

    def __init__(self, settings: Settings) -> None:
        """Initialize all feature interpolators.

        Args:
            settings: Application settings containing FPS and other parameters
        """
        # Initialize feature interpolators
        self._point_interpolater = PosePointInterpolater(settings)
        # Future: Add angle, measurement interpolators
        # self._angle_interpolater = PoseAngleInterpolater(settings)

        # Store latest interpolated results
        self._interpolated_poses: PoseDict = {}

    def add_poses(self, poses: PoseDict) -> None:
        """Add input samples to all interpolators (called at input rate)."""
        # Feed to all feature interpolators
        self._point_interpolater.add_poses(poses)
        # Future: self._angle_interpolater.add_poses(poses)

    def update(self, current_time: float | None = None) -> None:
        """Generate interpolated output for all poses (called at output rate)."""
        # Clear previous results
        self._interpolated_poses.clear()

        # Callback to collect point interpolation results
        def on_point_interpolation(poses: PoseDict) -> None:
            self._interpolated_poses.update(poses)

        # Register callback and update
        self._point_interpolater.add_callback(on_point_interpolation)
        self._point_interpolater.update(current_time)
        self._point_interpolater.remove_callback(on_point_interpolation)

        # Future: Chain additional interpolators
        # Current poses serve as input to next interpolator
        # if self._interpolated_poses:
        #     self._angle_interpolater.add_poses(self._interpolated_poses)
        #     self._angle_interpolater.update(current_time)

    def get_pose(self, pose_id: int) -> Optional[Pose]:
        """Get interpolated pose by ID."""
        return self._interpolated_poses.get(pose_id)

    def get_poses(self) -> PoseDict:
        """Get all currently interpolated poses."""
        return self._interpolated_poses

    def has_pose(self, pose_id: int) -> bool:
        """Check if a pose exists in current interpolation."""
        return pose_id in self._interpolated_poses

    def get_pose_ids(self) -> list[int]:
        """Get list of all available pose IDs."""
        return list(self._interpolated_poses.keys())
