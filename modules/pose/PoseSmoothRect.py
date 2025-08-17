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

        self.current_rect: Optional[Rect] = None

        # Spring-damper system state
        self.velocity: Rect = Rect(0.0, 0.0, 0.0, 0.0)
        self.spring_constant: float = 1000.0
        self.damping_ratio: float = 0.9

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


        self.spring_constant: float = 1000.0
        self.damping_ratio: float = 0.9

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

        # Apply spring-damper smoothing
        if self.current_rect is not None:
            smooth_x, self.velocity.x = self._apply_spring_damper(
                new_rect.x, self.current_rect.x, self.velocity.x,
                self.spring_constant, self.damping_ratio
            )
            smooth_y, self.velocity.y = self._apply_spring_damper(
                new_rect.y, self.current_rect.y, self.velocity.y,
                self.spring_constant, self.damping_ratio
            )
            smooth_h, self.velocity.height = self._apply_spring_damper(
                new_rect.height, self.current_rect.height, self.velocity.height,
                self.spring_constant, self.damping_ratio
            )
            smooth_w: float = smooth_h * self.dst_aspectratio
            self.current_rect = Rect(x=smooth_x, y=smooth_y, height=smooth_h, width=smooth_w)
        else:
            self.current_rect = new_rect

        # Notify callbacks with the updated rectangle
        self._notify_callback(pose, self.current_rect)

        if pose.is_final:
            self.current_rect = None
            self.velocity = Rect(0.0, 0.0, 0.0, 0.0)

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

    # STATIC METHODS
    @staticmethod
    def _apply_spring_damper(new_value: float, current_value: float,
                           velocity: float, spring_constant: float,
                           damping_ratio: float, dt: float = 1.0/60.0) -> tuple[float, float]:
        """Apply spring-damper physics to a single value. Returns (new_value, new_velocity)"""
        if current_value is None:
            return new_value, 0.0

        # Calculate spring force
        displacement = new_value - current_value
        spring_force = spring_constant * displacement

        # Calculate damping force
        damping_force = -2.0 * damping_ratio * np.sqrt(spring_constant) * velocity

        # Update velocity
        total_force = spring_force + damping_force
        new_velocity = velocity + total_force * dt

        # Update position
        result_value = current_value + new_velocity * dt

        return result_value, new_velocity