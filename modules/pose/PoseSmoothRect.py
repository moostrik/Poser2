# Standard library imports
from dataclasses import replace
from typing import Optional

# Third-party imports
import numpy as np

# Local application imports
from modules.pose.PoseDefinitions import *
from modules.pose.PoseDefinitions import Pose
from modules.tracker.Tracklet import Rect

from modules.utils.HotReloadMethods import HotReloadMethods

class PoseSmoothRect():
    def __init__(self, aspectratio: float = 9/16, smoothing_factor: float = 0.8) -> None:
        self.src_aspectratio: float = 16 / 9
        self.dst_aspectratio: float = aspectratio
        self.rect_output_callbacks: set[PoseCallback] = set()
        self.callback_lock: Lock = Lock()

        # Smoothing properties
        self.smoothing_factor: float = smoothing_factor
        self.current_rect: Optional[Rect] = None

        # Target relative positions
        self.nose_dest_x: float = 0.5   # Nose centered horizontally
        self.nose_dest_y: float = 0.4
        self.bottom_dest_y: float = 0.9  # Lowest ankle at 90% from the top

        hot_reload = HotReloadMethods(self.__class__, True, True)

    def _update(self, pose: Pose) -> None:

        self.smoothing_factor: float = 0.1
        self.nose_dest_x: float = 0.5
        self.nose_dest_y: float = 0.4
        self.bottom_dest_y: float = 0.95
        self.src_aspectratio: float = 16 / 9
        self.dst_aspectratio: float = 9 / 16

        # Get the bottom and nose positions
        pose_rect: Rect | None = pose.crop_rect
        if pose_rect is None:
            self._notify_callback(pose, self.current_rect)
            return
        bottom: float = min(pose_rect.bottom, 1.0)

        keypoints: np.ndarray | None = pose.get_absolute_keypoints()
        if pose.points is None or keypoints is None:
            self._notify_callback(pose, self.current_rect)
            return
        scores: np.ndarray | None = pose.points.scores
        nose_score: float = scores[Keypoint.nose.value]
        if nose_score < 0.3:
            self._notify_callback(pose, self.current_rect)
            return
        nose_x: float = keypoints[Keypoint.nose.value][0]
        nose_y: float = keypoints[Keypoint.nose.value][1]

        # Calculate rectangle dimensions
        height: float = (bottom - nose_y) / (self.bottom_dest_y- self.nose_dest_y)
        width: float = height * self.dst_aspectratio
        left: float = nose_x - width * self.nose_dest_x
        top: float = nose_y - height * self.nose_dest_y
        new_rect = Rect(left, top, width, height)

        # Apply smoothing
        x_smooth: float = self.smoothing_factor
        y_smooth: float = self.smoothing_factor / self.src_aspectratio

        if self.current_rect is not None:
            s_x: float = (1 - x_smooth) * self.current_rect.x + x_smooth * new_rect.x
            s_y: float = (1 - y_smooth) * self.current_rect.y + y_smooth * new_rect.y
            s_h: float = (1 - y_smooth) * self.current_rect.height + y_smooth * new_rect.height
            s_w: float = s_h * self.dst_aspectratio
            smooth_rect = Rect(x=s_x, y=s_y, height=s_h, width=s_w)
            self.current_rect = smooth_rect
        else:
            self.current_rect = new_rect

        # self.current_rect = new_rect

        # Notify callbacks with the updated rectangle
        self._notify_callback(pose, self.current_rect)

        if pose.is_final:
            # Reset current_rect when the pose is final
            self.current_rect = None

    def pose_input(self, pose: Pose) -> None:
        """Add a pose to the processing queue"""
        self._update(pose)

    def add_pose_callback(self, callback: PoseCallback) -> None:
        """Add callback for processed rectangles"""
        with self.callback_lock:
            self.rect_output_callbacks.add(callback)

    def _notify_callback(self, pose: Pose, rect: Rect | None) -> None:

        new_pose: Pose = replace(pose, smooth_rect=rect)
        """Handle processed rectangle"""
        with self.callback_lock:
            for callback in self.rect_output_callbacks:
                callback(new_pose)

