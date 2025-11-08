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
from modules.pose.filters import PoseAngleChaseInterpolator, PoseChaseInterpolatorConfig
from modules.pose.Pose import Pose, PoseDict

from modules.Settings import Settings
from modules.utils.PointsAndRects import Rect

from modules.utils.HotReloadMethods import HotReloadMethods

class RenderDataHub:
    def __init__(self, settings: Settings) -> None:
        self._num_players: int = settings.num_players

        self.interpolator_config: PoseChaseInterpolatorConfig = PoseChaseInterpolatorConfig(
            input_frequency=settings.camera_fps,
            responsiveness=0.2,
            friction=0.03
        )

        self.interpolators: dict[int, PoseAngleChaseInterpolator] = {
            i: PoseAngleChaseInterpolator(self.interpolator_config) for i in range(self._num_players)
        }

        # Lock to ensure thread safety
        self._lock = Lock()

        self._hot_reload = HotReloadMethods(self.__class__, True, True)

    def update(self) -> None:
        """Update all active trackers and smoothers."""
        for interpolator in self.interpolators.values():
            interpolator.update()

    def reset(self) -> None:
        """Reset all trackers and smoothers."""
        for interpolator in self.interpolators.values():
            interpolator.reset()

    def add_poses(self, poses: PoseDict) -> None:
        """ Add a new pose data point for processing."""
        for id, pose in poses.items():
            if not id in self.interpolators:
                self.interpolators[id] = PoseAngleChaseInterpolator(self.interpolator_config)
            self.interpolators[id].process(pose)
            if pose.lost:
                del self.interpolators[id]


    # ACTIVE
    def get_is_active(self, pose_id: int) -> bool: # for backward compatibility
        return self.has_pose(pose_id)

    def has_pose(self, pose_id: int) -> bool:
        """Check if a pose exists for the specified tracklet ID."""
        with self._lock:
            return pose_id in self.interpolators

    def get_pose(self, pose_id: int) -> Pose:
        """Get smoothed pose for the specified tracklet ID."""
        with self._lock:
            if pose_id not in self.interpolators:
                raise KeyError(f"No pose found for tracklet ID {pose_id}")
        return self.interpolators[pose_id].get_interpolated_pose()

    #  BODY JOINT ANGLES
    def get_angles(self, tracklet_id: int) -> PoseAngleData:
        """Get angles for the specified tracklet ID"""
        return self.get_pose(tracklet_id).angle_data

    def get_delta(self, tracklet_id: int) -> PoseAngleData:
        """Get delta for the specified tracklet ID"""
        return self.get_pose(tracklet_id).delta_data

    def get_symmetries(self, tracklet_id: int) -> PoseAngleSymmetryData:
        return self.get_pose(tracklet_id).symmetry_data

    # TIME
    def get_cumulative_motion(self, tracklet_id: int) -> float: # for backward compatibility
        return self.get_motion_time(tracklet_id)

    def get_motion_time(self, tracklet_id: int) -> float:
        """Get motion time for the specified tracklet ID."""
        return self.get_pose(tracklet_id).motion_time

    def get_age(self, tracklet_id: int) -> float:
        """Get the age in seconds since the tracklet was first detected."""
        return self.get_pose(tracklet_id).age
