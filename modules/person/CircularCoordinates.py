
from modules.person.Person import Person, PersonDict, CamTracklet, Tracklet, Rect, Point3f, AnglePosition
import numpy as np

CAMERA_FOV:float = 120.0
TARGET_FOV:float = 90.0

ANGLE_RANGE:float = 10
VERTICAL_RANGE:float = 0.05
SIZE_RANGE:float = 0.05

class CircularCoordinates():
    def __init__(self, num_cameras: int) -> None:
        self.num_cameras: int   = num_cameras
        self.cam_fov: float     = CAMERA_FOV
        self.target_fov: float  = 90.0

        self.k1: float = 0.05  # Distortion coefficient k1
        self.k2: float = 0.0  # Distortion coefficient k2

        self.angle_range: float     = ANGLE_RANGE
        self.vertical_range: float  = VERTICAL_RANGE
        self.size_range: float      = SIZE_RANGE

    def calc_angle_position(self, person: Person) -> AnglePosition:
        x_angle: float = self._calc_angle(person)
        y_pos: float = person.tracklet.roi.y
        size: float = max(person.tracklet.roi.height, person.tracklet.roi.width) / 10.0
        return AnglePosition(x_angle, y_pos, size)

    def _calc_angle(self, person: Person) -> float:
        normalized_x: float     = person.tracklet.roi.x * 2.0 - 1.0
        undistorted_x: float    = self.undistort_x(normalized_x, self.k1, self.k2)
        angle_overlap: float    = (self.cam_fov - self.target_fov) / 2.0
        local_angle: float      = undistorted_x * (self.cam_fov / 2) - angle_overlap + self.target_fov / 2.0
        world_angle: float      = self.target_fov * person.cam_id + local_angle
        normalized_angle: float = world_angle / 360.0
        return normalized_angle

    @staticmethod
    def undistort_x(x: float, k1: float, k2: float) -> float:
        return x + k1 * x**3 + k2 * x**5








    def in_range(self, AP1: AnglePosition, AP2: AnglePosition) -> bool:
        normalized_angle_range: float = self.angle_range / 360.0
        if abs(AP1.x_angle - AP2.x_angle) > normalized_angle_range:
            return False
        if abs(AP1.y_pos - AP2.y_pos) > self.vertical_range:
            return False
        if abs(AP1.size - AP2.size) > self.size_range:
            return False
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