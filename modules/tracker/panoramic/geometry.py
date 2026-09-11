import math

from modules.utils import Rect


# The play zone with its hard floor, as a distance from a camera: Ø 2 m to Ø 7 m is roughly
# 0.6 m to 4.0 m out. Clamping here means a mangled bounding box can only move the parallax
# correction within the band people are actually in, never to a nonsensical depth.
_MIN_DISTANCE: float = 0.6
_MAX_DISTANCE: float = 4.0


class Geometry:
    """Turns a camera's bounding box into a world azimuth.

    Two properties of the delivered frame make this simple, and both are produced by the
    camera's warp (`modules/oak/camera/definitions.py`, `equirect_mesh_points`), not assumed
    here: the frame is **level** (the tilt is undone) and **equirectangular** (a column is one
    azimuth at every height, a row is one elevation at every column). On the raw fisheye neither
    holds — a standing person's box centre reads several degrees short of their true bearing,
    worst near the seams — which is why there is no distortion correction in this class: the
    projection is fixed upstream rather than patched here.
    """

    def __init__(self, num_cameras: int, cam_fov: float, target_fov: float) -> None:
        self.num_cameras: int = num_cameras
        self.cam_fov: float = cam_fov
        self.target_fov: float = target_fov
        self.fov_overlap: float = (self.cam_fov - self.target_fov) / 2.0

        # Parallax: cameras sit on a ring of this radius (m), not at the shared
        # centre the world-angle model assumes. 0 disables the correction.
        self._ring_radius: float = 0.0
        # Lens height above the floor (m) — the one measured constant the distance estimate needs.
        self._camera_height: float = 0.5
        self._vfov: float = 79.4

    def get_angles_and_overlap(self, roi: Rect, cam_id: int, expansion: float) -> tuple[float, float, bool, float]:
        local_angle, world_angle, distance = self.calc_angle(roi, cam_id)
        overlap: bool = self.angle_in_overlap(local_angle, expansion)
        return (local_angle, world_angle, overlap, distance)

    def calc_angle(self, roi: Rect, cam_id: int) -> tuple[float, float, float]:
        local_angle: float = self._calc_local_angle(roi)
        distance: float = self.estimate_distance(roi)
        # Edge/overlap/hysteresis tests stay in the raw camera frame; only the
        # world angle is re-projected to the shared centre for cross-camera fusion.
        corrected_local: float = self._parallax_corrected_local(local_angle, distance)
        world_angle: float = self._calc_world_angle(corrected_local, cam_id)
        return local_angle, world_angle, distance

    def estimate_distance(self, roi: Rect) -> float:
        """Distance from the camera (m), from where the feet meet the floor.

        The frame is equirectangular and level, so a row *is* an elevation: the box's bottom edge
        is a depression angle below the horizon, and the floor plane turns that into a distance
        with one measured constant, the lens height. Nothing about the person enters it — arms
        raised, legs pulled up and bending over all change a box's *height*, and none of them
        move the feet.

        TWO CAMERA FACTS, HANDLED IN TWO DIFFERENT PLACES. The lens *height* is here, as
        ``camera_height``. The *tilt* is not: this assumes the frame's centre row is the horizon,
        which is true only because the camera's warp has already levelled it
        (``equirect_mesh_points``). That assumption cannot be checked from here and it is not a
        small one — on an un-levelled frame from a camera aimed up 15 deg, a person truly 3.26 m
        away reads as 1.14 m. So the distance, and therefore the parallax correction that depends
        on it, is meaningless on footage that has not been through the warp.

        The frame's own geometry bounds this nicely: its bottom row sits at about 39.7° of
        depression, which is 0.60 m out, just inside the Ø 2.0 m hard floor.

        **A box may extend outside the frame.** The device tracker extrapolates the extent of a
        partly-visible person, and nothing clamps it on the way in (`Tracklet.from_depthcam`), so
        a close person's `bottom` legitimately exceeds 1.0 and that is real information about how
        close they are. It is used, not discarded: the formula is continuous across the frame edge
        and the final clamp is what bounds the answer. (At `bottom == 1.0` the formula already
        gives 0.60 m, so there was never a boundary to special-case.) Past 90° of depression the
        tangent turns negative, which the clamp also catches.
        """
        bottom: float = roi.y + roi.height
        depression: float = math.radians((bottom - 0.5) * self._vfov)
        if depression <= 0.0:
            # Feet at or above the horizon: not standing on this floor.
            return _MAX_DISTANCE
        distance: float = self._camera_height / math.tan(depression)
        return max(_MIN_DISTANCE, min(_MAX_DISTANCE, distance))

    def _parallax_corrected_local(self, local_angle: float, distance: float) -> float:
        """Re-project a local angle so it reads as if seen from the rig centre.

        The camera faces radially outward, so the centre sits ``ring_radius``
        behind it. Placing the person at (distance, angle) in the camera frame
        and re-measuring the bearing from the centre removes the cross-camera
        seam disagreement caused by the off-centre mounting."""
        if self._ring_radius <= 0.0:
            return local_angle
        theta: float = math.radians(local_angle - self.cam_fov / 2.0)
        x: float = distance * math.cos(theta) + self._ring_radius
        y: float = distance * math.sin(theta)
        return math.degrees(math.atan2(y, x)) + self.cam_fov / 2.0

    def _calc_local_angle(self, roi: Rect) -> float:
        """The bearing of the box centre within this camera's field. Exact, not approximate:
        the frame is equirectangular, so a column is one azimuth at every height."""
        normalized_x: float = roi.x + roi.width / 2.0
        return normalized_x * self.cam_fov

    def _calc_world_angle(self, local_angle: float, cam_id: int) -> float:
        world_angle: float = self.target_fov * cam_id + local_angle - self.fov_overlap
        world_angle = world_angle % 360.0  # Ensure the angle is within 0 to 360 degrees
        return world_angle

    def angle_in_overlap(self, local_angle: float, expansion: float = 0.0) -> bool:
        angle_overlap: float = self.fov_overlap * (1.0 + expansion)

        if local_angle <= angle_overlap or local_angle >= self.cam_fov - angle_overlap:
            return True
        return False

    def angle_in_edge(self, local_angle: float, range_: float = 1.0) -> bool:
        edge: float = self.fov_overlap * range_

        if local_angle <= edge or local_angle >= self.cam_fov - edge:
            return True
        return False

    def angle_from_edge(self, local_angle: float) -> float:
        return min(local_angle, self.cam_fov - local_angle)

    @staticmethod
    def angle_diff(a: float, b: float) -> float:
        diff: float = abs(a - b)
        if diff > 180.0:
            diff = 360.0 - diff
        return diff

    # SET
    def set_fov(self, cam_fov: float) -> None:
        self.cam_fov = cam_fov
        self.fov_overlap = (self.cam_fov - self.target_fov) / 2.0

    def set_ring_radius(self, ring_radius: float) -> None:
        self._ring_radius = ring_radius

    def set_camera_height(self, camera_height: float) -> None:
        self._camera_height = camera_height

    def set_vfov(self, vfov: float) -> None:
        self._vfov = vfov
