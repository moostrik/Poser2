
from modules.person.Person import Person, PersonDict, Tracklet, Rect, Point3f
import numpy as np

from modules.person.Definitions import *

class CircularCoordinates():
    def __init__(self, num_cameras: int) -> None:
        self.num_cameras: int   = num_cameras
        self.cam_fov: float     = CAMERA_FOV
        self.target_fov: float  = 90
        self.fov_overlap: float = (self.cam_fov - self.target_fov) / 2.0

        self.k1: float = 0.05  # Distortion coefficient k1
        self.k2: float = 0.0  # Distortion coefficient k2

        self.angle_range: float     = ANGLE_RANGE / 360.0
        self.vertical_range: float  = VERTICAL_RANGE
        self.size_range: float      = SIZE_RANGE

    def calc_angles(self, persons: list[Person]) -> None:
        for person in persons:
            person.local_angle = self._calc_local_angle(person.tracklet.roi)
            person.world_angle = self._calc_world_angle(person.local_angle, person.cam_id)

    def _calc_local_angle(self, roi: Rect) -> float:
        normalized_x: float     = roi.x + roi.width / 2.0
        local_angle: float      = normalized_x * self.cam_fov
        return local_angle

    def _calc_world_angle(self, local_angle: float, cam_id: int) -> float:
        wold_angle: float = self.target_fov * cam_id + local_angle - self.fov_overlap
        world_angle: float = wold_angle % 360.0  # Ensure the angle is within 0 to 360 degrees
        if world_angle < 0:
            world_angle += 360.0
        return world_angle

    def angle_in_overlap(self, world_angle: float, range: float = 1.0) -> bool:
        angle_overlap: float    = self.fov_overlap * range
        local_angle: float      = world_angle % self.target_fov

        if local_angle <= angle_overlap or local_angle >= self.target_fov - angle_overlap:
            return True
        return False

    def angle_in_edge(self, local_angle: float, range: float = 1.0) -> bool:
        edge: float    = self.fov_overlap * range

        if local_angle <= edge or local_angle >= self.cam_fov - edge:
            return True
        return False


    @staticmethod
    def undistort_x(x: float, k1: float, k2: float) -> float:
        return x
        return 0.5 * (1 + np.tanh(k1 * (2*x - 1) + k2 * (2*x - 1)**3))

    # SET
    def set_fov(self, cam_fov: float) -> None:
        self.cam_fov = cam_fov
        self.fov_overlap: float = (self.cam_fov - self.target_fov) / 2.0

    def set_angle_range(self, angle_range: float) -> None:
        self.angle_range = angle_range / 360.0

    def set_vertical_range(self, vertical_range: float) -> None:
        self.vertical_range = vertical_range

    def set_size_range(self, size_range: float) -> None:
        self.size_range = size_range
