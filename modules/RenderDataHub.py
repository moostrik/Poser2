"""
Provides smoothed pose data for rendering at output framerate.

Manages pose tracking components that smooth noisy input and interpolate between
samples, enabling higher framerate output than input (e.g., 24 FPS input → 60 FPS output).

Counterpart to CaptureDataHub:
- CaptureDataHub: Stores raw pose data at capture rate (input FPS)
- RenderDataHub: Provides smoothed pose data at render rate (output FPS)

Key capabilities:
1. Coordinates viewport, angle, and correlation tracking
2. Provides thread-safe access to smoothed pose data
3. Enables higher framerate output through interpolation
4. Manages pose correlation analysis between tracklets

Note: Interpolation introduces latency of approximately 1 input frame.
"""

# Standard library imports
from threading import Lock

# Local application imports
from modules.pose.features.PoseAngles import PoseAngleData
from modules.pose.features.PoseAngleSymmetry import PoseAngleSymmetryData
from modules.pose.filters import PoseInterpolator
from modules.pose.Pose import Pose, PoseDict

from modules.Settings import Settings
from modules.utils.PointsAndRects import Rect

from modules.utils.HotReloadMethods import HotReloadMethods

class RenderDataHub:
    def __init__(self, settings: Settings) -> None:
        self._num_players: int = settings.num_players

        self.interpolator = PoseInterpolator(settings)
        self.poses: PoseDict = {}

        # Lock to ensure thread safety
        self._lock = Lock()

        self._hot_reload = HotReloadMethods(self.__class__, True, True)

    def update(self) -> None:
        """Update all active trackers and smoothers."""
        self.interpolator.update()
        with self._lock:
            self.poses = self.interpolator.get_poses()

    def reset(self) -> None:
        """Reset all trackers and smoothers."""
        self.interpolator.reset()

    def add_poses(self, poses: PoseDict) -> None:
        """ Add a new pose data point for processing."""
        self.interpolator.add_poses(poses)

    # ACTIVE
    def get_is_active(self, tracklet_id: int) -> bool: # for backward compatibility
        return self.has_pose(tracklet_id)

    def has_pose(self, tracklet_id: int) -> bool:
        """Check if a pose exists for the specified tracklet ID."""
        with self._lock:
            pose: Pose | None = self.poses.get(tracklet_id)
            if pose is None:
                return False
        return True

    def get_pose(self, tracklet_id: int) -> Pose:
        """Get smoothed pose for the specified tracklet ID."""
        with self._lock:
            pose: Pose | None = self.poses.get(tracklet_id)
            if pose is None:
                raise KeyError(f"No pose found for tracklet ID {tracklet_id}")
        return pose

    #  BODY JOINT ANGLES
    def get_angles(self, tracklet_id: int) -> PoseAngleData:
        """Get angles for the specified tracklet ID"""
        return self.get_pose(tracklet_id).angle_data

    def get_delta(self, tracklet_id: int) -> PoseAngleData:
        """Get delta for the specified tracklet ID"""
        return self.get_pose(tracklet_id).delta_data

    def get_symmetries(self, tracklet_id: int) -> PoseAngleSymmetryData:
        return self.get_pose(tracklet_id).similarity_data

    # TIME
    def get_cumulative_motion(self, tracklet_id: int) -> float: # for backward compatibility
        return self.get_motion_time(tracklet_id)

    def get_motion_time(self, tracklet_id: int) -> float:
        """Get motion time for the specified tracklet ID."""
        return self.get_pose(tracklet_id).motion_time

    def get_age(self, tracklet_id: int) -> float:
        """Get the age in seconds since the tracklet was first detected."""
        return self.get_pose(tracklet_id).age
