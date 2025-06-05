
from modules.person.Person import Person, PersonDict, Tracklet, Rect, Point3f, AnglePosition
import numpy as np

from modules.person.Definitions import *

class CircularCoordinates():
    def __init__(self, num_cameras: int) -> None:
        self.num_cameras: int   = num_cameras
        self.cam_fov: float     = CAMERA_FOV
        self.target_fov: float  = 90

        self.k1: float = 0.05  # Distortion coefficient k1
        self.k2: float = 0.0  # Distortion coefficient k2

        self.angle_range: float     = ANGLE_RANGE / 360.0
        self.vertical_range: float  = VERTICAL_RANGE
        self.size_range: float      = SIZE_RANGE

    def calc_angles(self, persons: list[Person]) -> None:
        for person in persons:
            person.angle = self._calc_angle(person)

    def _calc_angle(self, person: Person) -> float:
        normalized_x: float     = person.tracklet.roi.x
        undistorted_x: float    = self.undistort_x(normalized_x, self.k1, self.k2)
        angle_overlap: float    = (self.cam_fov - self.target_fov) / 2.0
        local_angle: float      = undistorted_x * (self.cam_fov) - angle_overlap
        world_angle: float      = self.target_fov * person.cam_id + local_angle
        normalized_angle: float = (world_angle % 360)
        return normalized_angle

    def angle_in_overlap(self, angle: float, range: float) -> bool:
        angle_overlap: float    = (self.cam_fov - self.target_fov) * 0.5 * range
        local_angle: float      = angle % self.target_fov
        if local_angle < angle_overlap or local_angle > self.target_fov - angle_overlap:
            return True
        return False


    @staticmethod
    def undistort_x(x: float, k1: float, k2: float) -> float:
        return x
        return 0.5 * (1 + np.tanh(k1 * (2*x - 1) + k2 * (2*x - 1)**3))

    # SET
    def set_fov(self, cam_fov: float) -> None:
        self.cam_fov = cam_fov

    def set_angle_range(self, angle_range: float) -> None:
        self.angle_range = angle_range / 360.0

    def set_vertical_range(self, vertical_range: float) -> None:
        self.vertical_range = vertical_range

    def set_size_range(self, size_range: float) -> None:
        self.size_range = size_range
