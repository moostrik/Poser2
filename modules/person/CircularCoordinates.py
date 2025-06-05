
from modules.person.Person import Person, PersonDict, CamTracklet, Tracklet, Rect, Point3f, AnglePosition
import numpy as np

from modules.person.Definitions import *

class CircularCoordinates():
    def __init__(self, num_cameras: int) -> None:
        self.num_cameras: int   = num_cameras
        self.cam_fov: float     = CAMERA_FOV
        self.target_fov: float  = 90.0

        self.k1: float = 0.05  # Distortion coefficient k1
        self.k2: float = 0.0  # Distortion coefficient k2

        self.angle_range: float     = ANGLE_RANGE / 360.0
        self.vertical_range: float  = VERTICAL_RANGE
        self.size_range: float      = SIZE_RANGE

    def calc_angle_position(self, person: Person) -> AnglePosition:
        x_angle: float = self._calc_angle(person)
        y_pos: float = person.tracklet.roi.y
        size: float = max(person.tracklet.roi.height, person.tracklet.roi.width)
        return AnglePosition(x_angle, y_pos, size)

    def _calc_angle(self, person: Person) -> float:
        normalized_x: float     = person.tracklet.roi.x
        undistorted_x: float    = self.undistort_x(normalized_x, self.k1, self.k2)
        angle_overlap: float    = (self.cam_fov - self.target_fov) / 2.0
        local_angle: float      = undistorted_x * (self.cam_fov) - angle_overlap
        world_angle: float      = self.target_fov * person.cam_id + local_angle
        normalized_angle: float = (world_angle % 360) / 360.0
        return normalized_angle

    @staticmethod
    def undistort_x(x: float, k1: float, k2: float) -> float:
        return x
        return 0.5 * (1 + np.tanh(k1 * (2*x - 1) + k2 * (2*x - 1)**3))

    def in_range(self, AP1: AnglePosition, AP2: AnglePosition) -> bool:
        angle_diff: float = abs(AP1.x_angle - AP2.x_angle)
        if min(angle_diff, 1.0 - angle_diff) > self.angle_range:
            return False
        if abs(AP1.y_pos - AP2.y_pos) > self.vertical_range:
            return False
        # if abs(AP1.size - AP2.size) > self.size_range:
        #     return False
        return True

    def distance(self, AP1: AnglePosition, AP2: AnglePosition) -> float:
        return ((AP1.x_angle - AP2.x_angle) ** 2 + (AP1.y_pos - AP2.y_pos) ** 2 + (AP1.size - AP2.size) ** 2) ** 0.5

    def find(self, person: Person, person_dict: PersonDict) -> Person | None:
        person_list: list[Person] = []
        for key in person_dict.keys():
            if self.in_range(person.angle_pos, person_dict[key].angle_pos):
                person_list.append(person_dict[key])

        person_list.sort(key=lambda p: self.distance(person.angle_pos, p.angle_pos))
        if len(person_list) > 0:
            return person_list[0]

        return None

    # SET
    def set_fov(self, cam_fov: float) -> None:
        self.cam_fov = cam_fov

    def set_angle_range(self, angle_range: float) -> None:
        self.angle_range = angle_range / 360.0

    def set_vertical_range(self, vertical_range: float) -> None:
        self.vertical_range = vertical_range

    def set_size_range(self, size_range: float) -> None:
        self.size_range = size_range
