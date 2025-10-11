from threading import Lock

from modules.pose.Pose import Pose
from modules.pose.PoseTypes import PoseJoint
from modules.pose.smooth.PoseSmoothRect import PoseSmoothRect, PoseSmoothRectSettings
from modules.pose.smooth.PoseSmoothAngles import PoseSmoothAngles, PoseSmoothAngleSettings
from modules.pose.smooth.PoseSmoothHead import PoseSmoothHead, PoseSmoothHeadSettings
from modules.utils.OneEuroInterpolation import OneEuroSettings
from modules.utils.PointsAndRects import Rect

from modules.utils.HotReloadMethods import HotReloadMethods

class PoseSmoothData:
    """
    Manages multiple PoseSmoothRect and PoseSmoothAngles instances for different poses.
    Uses tracklet IDs as keys to track different poses.
    """
    def __init__(self, num_players: int) -> None:
        self._num_players: int = num_players

        self.OneEuro_settings: OneEuroSettings = OneEuroSettings(25, 1.0, 0.1)
        self.rect_settings: PoseSmoothRectSettings = PoseSmoothRectSettings(
            smooth_settings=self.OneEuro_settings,
            center_dest_x=0.5,
            centre_dest_y=0.2,
            height_dest=0.95,
            src_aspectratio=16/9,
            dst_aspectratio=9/16
        )
        self.angle_settings: PoseSmoothAngleSettings = PoseSmoothAngleSettings(
            smooth_settings=self.OneEuro_settings,
            motion_threshold=0.002
        )
        self.head_settings: PoseSmoothHeadSettings = PoseSmoothHeadSettings(
            smooth_settings=self.OneEuro_settings,
            motion_threshold=0.002
        )

        # Dictionaries to store smoothers for each tracklet ID
        self._rect_smoothers: dict[int, PoseSmoothRect] = {}
        self._angle_smoothers: dict[int, PoseSmoothAngles] = {}
        self._head_smoothers: dict[int, PoseSmoothHead] = {}

        for i in range(num_players):
            self._rect_smoothers[i] = PoseSmoothRect(self.rect_settings)
            self._angle_smoothers[i] = PoseSmoothAngles(self.angle_settings)
            self._head_smoothers[i] = PoseSmoothHead(self.head_settings)

        # Lock to ensure thread safety
        self._lock = Lock()

        self._hot_reload = HotReloadMethods(self.__class__, True, True)

    def add_pose(self, pose: Pose) -> None:
        """ Add a new pose data point for processing."""
        with self._lock:
            tracklet_id: int = pose.tracklet.id
            self._rect_smoothers[tracklet_id].add_pose(pose)
            self._angle_smoothers[tracklet_id].add_pose(pose)
            self._head_smoothers[tracklet_id].add_pose(pose)

    def update(self) -> None:
        """Update all active smoothers."""
        with self._lock:
            for smoother in self._rect_smoothers.values():
                smoother.update()
            for smoother in self._angle_smoothers.values():
                smoother.update()
            for smoother in self._head_smoothers.values():
                smoother.update()

    def get_active_ids(self) -> list[int]:
        """Get a list of currently active tracklet IDs."""
        with self._lock:
            return [tracklet_id for tracklet_id, smoother in self._rect_smoothers.items() if smoother.is_active]

    def get_is_active(self, tracklet_id: int) -> bool:
        """Check if the smoother for the specified tracklet ID is active."""
        with self._lock:
            return self._rect_smoothers[tracklet_id].is_active

    def get_rect(self, tracklet_id: int) -> Rect:
        """Get smoothed rectangle for the specified tracklet ID."""
        with self._lock:
            return self._rect_smoothers[tracklet_id].smoothed_rect

    def get_angles(self, tracklet_id: int) -> dict[PoseJoint, float]:
        """Get smoothed angles for all joints for the specified tracklet ID."""
        with self._lock:
            return self._angle_smoothers[tracklet_id].angles

    def get_angle(self, tracklet_id: int, joint: PoseJoint) -> float:
        """Get smoothed angle for the specified tracklet ID and joint."""
        with self._lock:
            return self._angle_smoothers[tracklet_id].get_angle(joint, symmetric=True)

    def get_deltas(self, tracklet_id: int) -> dict[PoseJoint, float]:
        """Get smoothed angle changes for all joints for the specified tracklet ID."""
        with self._lock:
            return self._angle_smoothers[tracklet_id].deltas

    def get_delta(self, tracklet_id: int, joint: PoseJoint) -> float:
        """Get smoothed angle for the specified tracklet ID and joint."""
        with self._lock:
            return self._angle_smoothers[tracklet_id].get_delta(joint, symmetric=True)

    def get_angular_motion(self, tracklet_id: int) -> float:
        """Get smoothed angle change for the specified tracklet ID and joint."""
        with self._lock:
            return self._angle_smoothers[tracklet_id].total_motion

    def get_head(self, tracklet_id: int) -> float:
        """Get smoothed head angles for the specified tracklet ID."""
        with self._lock:
            return self._head_smoothers[tracklet_id].orientation

    def get_head_delta(self, tracklet_id: int) -> float:
        """Get smoothed head angle changes for the specified tracklet ID."""
        with self._lock:
            return self._head_smoothers[tracklet_id].delta

    def get_head_motion(self, tracklet_id: int) -> float:
        """Get smoothed head motion for the specified tracklet ID."""
        with self._lock:
            return self._head_smoothers[tracklet_id].motion

    def get_motion(self, tracklet_id: int) -> float:
        """Get combined motion (body + head) for the specified tracklet ID."""
        with self._lock:
            body_motion: float = self._angle_smoothers[tracklet_id].total_motion
            head_motion: float = self._head_smoothers[tracklet_id].motion
            return body_motion + head_motion

    def get_age(self, tracklet_id: int) -> float:
        """Get the age in seconds since the tracklet was first detected."""
        with self._lock:
            return self._rect_smoothers[tracklet_id].age

    def reset(self) -> None:
        """Reset all smoothers for all tracklet IDs."""
        with self._lock:
            for smoother in self._rect_smoothers.values():
                smoother.reset()
            for smoother in self._angle_smoothers.values():
                smoother.reset()
            for smoother in self._head_smoothers.values():
                smoother.reset()