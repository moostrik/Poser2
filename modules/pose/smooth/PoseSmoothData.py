import numpy as np
from threading import Lock
from typing import Dict, Optional

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
    def __init__(self, angle_settings: OneEuroSettings, rect_settings: PoseSmoothRectSettings) -> None:
        self.angle_settings: OneEuroSettings = angle_settings
        self.rect_settings: PoseSmoothRectSettings = rect_settings

        # Dictionaries to store smoothers for each tracklet ID
        self.rect_smoothers: Dict[int, PoseSmoothRect] = {}
        self.angle_smoothers: Dict[int, PoseSmoothAngles] = {}

        # Lock to ensure thread safety
        self._lock = Lock()

        hot_reload = HotReloadMethods(self.__class__, True, True)

    def add_pose(self, pose: Pose) -> None:
        """
        Add a new pose data point for processing.
        Creates new smoothers if this is a new tracklet ID.
        """
        with self._lock:
            tracklet_id: int = pose.tracklet.id

            # Create new smoothers if this is a new tracklet ID
            if tracklet_id not in self.rect_smoothers:
                self.rect_smoothers[tracklet_id] = PoseSmoothRect(self.rect_settings)

            if tracklet_id not in self.angle_smoothers:
                self.angle_smoothers[tracklet_id] = PoseSmoothAngles(self.angle_settings)

            # Add pose data to the respective smoothers
            self.rect_smoothers[tracklet_id].add_pose(pose)
            self.angle_smoothers[tracklet_id].add_pose(pose)

    def get_smoothed_rect(self, tracklet_id: int) -> Optional[Rect]:
        """Get smoothed rectangle for the specified tracklet ID."""
        with self._lock:
            if tracklet_id in self.rect_smoothers:
                return self.rect_smoothers[tracklet_id].get()
            return None

    def get_smoothed_angle(self, tracklet_id: int, joint: PoseJoint, symmetric: bool = False) -> Optional[float]:
        """Get smoothed angle for the specified tracklet ID and joint."""
        with self._lock:
            if tracklet_id in self.angle_smoothers:
                return self.angle_smoothers[tracklet_id].get_smoothed_angle(joint, symmetric)
            return None

    def get_joint_symmetry(self, tracklet_id: int, joint_type) -> Optional[float]:
        """Get joint symmetry for the specified tracklet ID and joint type."""
        with self._lock:
            if tracklet_id in self.angle_smoothers:
                return self.angle_smoothers[tracklet_id].get_joint_symmetry(joint_type)
            return None

    def get_average_symmetry(self, tracklet_id: int) -> Optional[float]:
        """Get average symmetry for the specified tracklet ID."""
        with self._lock:
            if tracklet_id in self.angle_smoothers:
                return self.angle_smoothers[tracklet_id].get_average_symmetry()
            return None

    def reset(self) -> None:
        """Reset all smoothers for all tracklet IDs."""
        with self._lock:
            for smoother in self.rect_smoothers.values():
                smoother.reset()
            for smoother in self.angle_smoothers.values():
                smoother.reset()