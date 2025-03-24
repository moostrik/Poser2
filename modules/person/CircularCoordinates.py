
from modules.person.Person import Person, PersonDict, CamTracklet, Tracklet, Rect, Point3f
CAMERA_ANGLE:float = 120.0

ANGLE_RANGE:float = 10.0
# Y_RANGE:float = 0.1 # without stereo: normalised coordinates
# Z_RANGE:float = 0.1 # without stereo: normalised coordinates
Y_RANGE:float = 200 # with stereo: coordinates in mm
Z_RANGE:float = 200 # with stereo: coordinates in mm

class CircularCoordinates():
    def __init__(self, num_cameras: int) -> None:
        self.num_cameras: int = num_cameras
        self.angle: float = CAMERA_ANGLE
        self.overlap: float = (self.angle * num_cameras - 360.0) / 2.0

        self.angle_range: float = ANGLE_RANGE
        self.y_range: float = Y_RANGE
        self.z_range: float = Z_RANGE

    def get_angle(self, x: float, cam_id: int) -> float:
        return x * self.angle - self.overlap

    def add_angle_position(self, person: Person) -> None:
        tracklet: Tracklet = person.tracklet
        position: Point3f = Point3f(tracklet.roi.x, tracklet.roi.y, 0.0)
        # if tracklet.spatialCoordinates.z != 0:
        #     position = tracklet.spatialCoordinates
        person.angle_pos = position

    def in_range(self, AP1: Point3f, AP2: Point3f) -> bool:
        # if abs(AP1.x - AP2.x) < self.angle_range and abs(AP1.y - AP2.y) < self.y_range and abs(AP1.z - AP2.z) < self.z_range:
        if abs(AP1.x - AP2.x) < self.angle_range:
            return True
        return False

    def distance(self, AP1: Point3f, AP2: Point3f) -> float:
        # return ((AP1.x - AP2.x) ** 2 + (AP1.y - AP2.y) ** 2 + (AP1.z - AP2.z) ** 2) ** 0.5
        return AP1.x - AP2.x

    def find(self, person: Person, person_dict: PersonDict) -> Person | None:
        person_list: list[Person] = []
        for key in person_dict.keys():
            if self.in_range(person.angle_pos, person_dict[key].angle_pos):
                person_list.append(person_dict[key])

        person_list.sort(key=lambda p: self.distance(person.angle_pos, p.angle_pos))
        if len(person_list) > 0:
            return person_list[0]

        return None