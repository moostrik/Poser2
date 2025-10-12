# Standard library imports
import numpy as np
from dataclasses import dataclass

# Local imports
from modules.pose.Pose import Pose
from modules.pose.PoseTypes import PoseJoint
from modules.pose.smooth.PoseSmoothBase import PoseSmoothBase

from modules.utils.PointsAndRects import Rect
from modules.utils.OneEuroInterpolation import OneEuroInterpolator, OneEuroSettings

from modules.utils.HotReloadMethods import HotReloadMethods

@dataclass
class PoseSmoothRectSettings:
    smooth_settings: OneEuroSettings
    center_dest_x: float = 0.5
    centre_dest_y: float = 0.2
    height_dest: float = 0.95
    src_aspectratio: float = 16/9
    dst_aspectratio: float = 9/16

class PoseSmoothRect(PoseSmoothBase):
    def __init__(self, settings: PoseSmoothRectSettings) -> None:
        self._active: bool = False
        self._settings: PoseSmoothRectSettings = settings
        self._src_aspectratio: float = settings.src_aspectratio
        self._dst_aspectratio: float = settings.dst_aspectratio

        self._center_x_interpolator: OneEuroInterpolator = OneEuroInterpolator(self._settings.smooth_settings)
        self._center_y_interpolator: OneEuroInterpolator = OneEuroInterpolator(self._settings.smooth_settings)
        self._height_interpolator: OneEuroInterpolator = OneEuroInterpolator(self._settings.smooth_settings)

        self._smoothed_rect: Rect = Rect(0.0, 0.0, 1.0, 1.0)
        self._age: float = 0.0

        self._hot_reload = HotReloadMethods(self.__class__, True, True)

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def smoothed_rect(self) -> Rect:
        return self._smoothed_rect

    @property
    def age(self) -> float:
        return self._age

    def add_pose(self, pose: Pose) -> None:
        if pose.tracklet.is_removed:
            self._active = False
            self.reset()
            return

        if pose.tracklet.is_active and not self._active:
            self._active = True
            self.reset()

        if not self._active:
            return

        self._age = pose.tracklet.age_in_seconds

        pose_rect: Rect | None = pose.crop_rect
        pose_points: np.ndarray | None = pose.point_data.points if pose.point_data is not None else None
        pose_height: float | None = pose.measurement_data.length_estimate if pose.measurement_data is not None else None

        if pose_rect is None:
            print(f"PoseSmoothRect: No crop rect for pose {pose.tracklet.id}, this should not happen")
            return

        # Always add data, OneEuroInterpolator will handle missing data
        if pose_points is None or pose_height is None:
            self._center_x_interpolator.add_sample(np.nan)
            self._center_y_interpolator.add_sample(np.nan)
            self._height_interpolator.add_sample(np.nan)
            return

        left_eye: np.ndarray = pose_points[PoseJoint.left_eye.value]
        right_eye: np.ndarray = pose_points[PoseJoint.right_eye.value]
        eye_midpoint: np.ndarray = (left_eye + right_eye) / 2

        centre_x: float = eye_midpoint[0] * pose_rect.width + pose_rect.x
        centre_y: float = eye_midpoint[1] * pose_rect.height + pose_rect.y
        height: float = pose_height # * pose_rect.height -> this is already is based on height

        self._center_x_interpolator.add_sample(centre_x)
        self._center_y_interpolator.add_sample(centre_y)
        self._height_interpolator.add_sample(height)

    def update(self) -> None:
        if not self._active:
            return

        self._center_x_interpolator.update()
        self._center_y_interpolator.update()
        self._height_interpolator.update()

        nose_x: float | None = self._center_x_interpolator._smooth_value
        nose_y: float | None = self._center_y_interpolator._smooth_value
        height: float | None = self._height_interpolator._smooth_value

        if nose_x is None or nose_y is None or height is None:
            return None

        # Apply height adjustment factor
        height = height / self._settings.height_dest

        # Calculate base width for destination aspect ratio
        base_width: float = height * self._settings.dst_aspectratio

        # # Apply aspect ratio correction
        # # When converting from src_ar to dst_ar, we need to adjust width
        # # to maintain proper proportional representation
        # aspect_ratio_correction: float = self.settings.src_aspectratio / self.settings.dst_aspectratio

        # # Adjust width based on the relationship between source and destination aspect ratios
        # # This ensures proper scaling when converting between different aspect ratios
        # corrected_width: float = base_width * aspect_ratio_correction

        corrected_width = base_width

        # Calculate position, adjusting for the corrected width
        left: float = nose_x - corrected_width * self._settings.center_dest_x
        top: float = nose_y - height * self._settings.centre_dest_y

        self._smoothed_rect = Rect(left, top, corrected_width, height)

    def reset(self) -> None:
        self._center_x_interpolator.reset()
        self._center_y_interpolator.reset()
        self._height_interpolator.reset()
        self._smoothed_rect = Rect(0.0, 0.0, 1.0, 1.0)
        self._age = 0.0
