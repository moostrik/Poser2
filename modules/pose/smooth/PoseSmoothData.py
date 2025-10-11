from threading import Lock

from modules.pose.Pose import Pose
from modules.pose.PoseTypes import PoseJoint
from modules.pose.smooth.PoseSmoothRect import PoseSmoothRect, PoseSmoothRectSettings
from modules.pose.smooth.PoseSmoothAngles import PoseSmoothAngles
from modules.utils.OneEuroInterpolation import OneEuroSettings
from modules.utils.PointsAndRects import Rect

from modules.utils.HotReloadMethods import HotReloadMethods

class PoseSmoothData:
    """
    Manages multiple PoseSmoothRect and PoseSmoothAngles instances for different poses.
    Uses tracklet IDs as keys to track different poses.
    """
    def __init__(self, num_players: int, angle_settings: OneEuroSettings, rect_settings: PoseSmoothRectSettings) -> None:
        self.num_players: int = num_players
        self.angle_settings: OneEuroSettings = angle_settings
        self.rect_settings: PoseSmoothRectSettings = rect_settings

        # Dictionaries to store smoothers for each tracklet ID
        self.rect_smoothers: dict[int, PoseSmoothRect] = {}
        self.angle_smoothers: dict[int, PoseSmoothAngles] = {}

        for i in range(num_players):
            self.rect_smoothers[i] = PoseSmoothRect(self.rect_settings)
            self.angle_smoothers[i] = PoseSmoothAngles(self.angle_settings)

        # Lock to ensure thread safety
        self._lock = Lock()

        self._hot_reload = HotReloadMethods(self.__class__, True, True)

    def add_pose(self, pose: Pose) -> None:
        """ Add a new pose data point for processing."""
        with self._lock:
            tracklet_id: int = pose.tracklet.id
            self.rect_smoothers[tracklet_id].add_pose(pose)
            self.angle_smoothers[tracklet_id].add_pose(pose)

    def update(self) -> None:
        """Update all active smoothers."""
        with self._lock:
            for smoother in self.rect_smoothers.values():
                smoother.update()
            for smoother in self.angle_smoothers.values():
                smoother.update()

    def get_active_ids(self) -> list[int]:
        """Get a list of currently active tracklet IDs."""
        with self._lock:
            return [tracklet_id for tracklet_id, smoother in self.rect_smoothers.items() if smoother.is_active]

    def get_is_active(self, tracklet_id: int) -> bool:
        """Check if the smoother for the specified tracklet ID is active."""
        with self._lock:
            return self.rect_smoothers[tracklet_id].is_active

    def get_rect(self, tracklet_id: int) -> Rect:
        """Get smoothed rectangle for the specified tracklet ID."""
        with self._lock:
            return self.rect_smoothers[tracklet_id].smoothed_rect

    def get_angles(self, tracklet_id: int) -> dict[PoseJoint, float]:
        """Get smoothed angles for all joints for the specified tracklet ID."""
        with self._lock:
            return self.angle_smoothers[tracklet_id].angles

    def get_angle(self, tracklet_id: int, joint: PoseJoint) -> float:
        """Get smoothed angle for the specified tracklet ID and joint."""
        with self._lock:
            return self.angle_smoothers[tracklet_id].get_angle(joint, symmetric=True)

    def get_deltas(self, tracklet_id: int) -> dict[PoseJoint, float]:
        """Get smoothed angle changes for all joints for the specified tracklet ID."""
        with self._lock:
            return self.angle_smoothers[tracklet_id].deltas

    def get_delta(self, tracklet_id: int, joint: PoseJoint) -> float:
        """Get smoothed angle for the specified tracklet ID and joint."""
        with self._lock:
            return self.angle_smoothers[tracklet_id].get_delta(joint, symmetric=True)

    def get_angular_motion(self, tracklet_id: int) -> float:
        """Get smoothed angle change for the specified tracklet ID and joint."""
        with self._lock:
            return self.angle_smoothers[tracklet_id].total_motion

    def reset(self) -> None:
        """Reset all smoothers for all tracklet IDs."""
        with self._lock:
            for smoother in self.rect_smoothers.values():
                smoother.reset()
            for smoother in self.angle_smoothers.values():
                smoother.reset()