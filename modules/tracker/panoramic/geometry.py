import math

from modules.oak import FrameWindow, frame_window
from modules.utils import Rect


# The play zone with its hard floor, as a distance from a camera: Ø 2 m to Ø 7 m is roughly
# 0.6 m to 4.0 m out. Clamping here means a mangled bounding box can only move the parallax
# correction within the band people are actually in, never to a nonsensical depth.
_MIN_DISTANCE: float = 0.6
_MAX_DISTANCE: float = 4.0

# Tallest reading `estimate_height` will report (m). A box whose feet land a pixel below the
# horizon divides by almost nothing, so the ratio needs a ceiling. Set above anything a person
# can measure — a tall one with both arms up reaches ~2.4 — so the cap only ever catches a
# mangled box, never a real reading it might otherwise flatten.
_MAX_HEIGHT: float = 3.0


class Geometry:
    """Turns a camera's bounding box into a world azimuth.

    Two properties of the delivered frame make this simple, and both are produced by the
    camera's warp (`modules/oak/camera/definitions.py`, `warp_mesh_points`), not assumed
    here: the frame is **level** (the tilt is undone) and **cylindrical** (a column is one
    azimuth at every height, a row is one elevation at every column, spaced by the tangent of
    the elevation). On the raw fisheye neither holds — a standing person's box centre reads
    several degrees short of their true bearing, worst near the seams — which is why there is no
    distortion correction in this class: the projection is fixed upstream rather than patched
    here. The row model (`FrameWindow`) is handed in by `set_window`.
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
        # The frame's rows: where the horizon is (px) and how many px a unit of tangent spans.
        # Until `set_window`, an untilted 1280 x 800 frame with the ideal lens.
        self.set_window(frame_window((1280, 800), (1280, 800), 1280, cam_fov, 0.0), 800)

    def get_angles_and_overlap(self, roi: Rect, cam_id: int,
                               expansion: float) -> tuple[float, float, bool, float, float]:
        """Everything one box says about one person, in the order `Annotation` holds it:
        local angle, world angle, overlap flag, distance (m) and height (m)."""
        local_angle, world_angle, distance = self.calc_angle(roi, cam_id)
        overlap: bool = self.angle_in_overlap(local_angle, expansion)
        return (local_angle, world_angle, overlap, distance, self.estimate_height(roi))

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

        The frame is cylindrical and level, so the rows below the horizon are the TANGENT of the
        depression: `tan(depression) = (bottom_px - horizon_px) / focal`. The floor plane turns
        that into a distance with one measured constant, the lens height, and the tangent
        cancels: `distance = camera_height * focal / (bottom_px - horizon_px)`. Nothing about
        the person enters it — arms raised, legs pulled up and bending over all change a box's
        *height*, and none of them move the feet.

        TWO CAMERA FACTS, HANDLED IN TWO DIFFERENT PLACES. The lens *height* is here, as
        ``camera_height``. The *tilt* is not: it is inside the window (`set_window`), which the
        camera's warp and this class derive with the same function (`frame_window`). The horizon
        is NOT the centre row — the window is pinned at the sensor's bottom reach, so on a camera
        aimed up 15 deg the horizon sits at row 0.78 of the frame — and reading a row as if it
        were is how a person truly 3.3 m away used to read as 1.1 m. The distance, and therefore
        the parallax correction that depends on it, is meaningless on footage that has not been
        through the warp.

        The frame's own geometry bounds this: its bottom row is the sensor's lowest reach on the
        centre column, `-elevation_bottom` of depression — 1.37 m out at P720 and tilt 15,
        1.16 m at P800 and tilt 16 — and anyone closer has their feet below the frame.

        **A box may extend outside the frame.** The device tracker extrapolates the extent of a
        partly-visible person, and nothing clamps it on the way in (`Tracklet.from_depthcam`), so
        a close person's `bottom` legitimately exceeds 1.0 and that is real information about how
        close they are. It is used, not discarded: the formula is continuous across the frame edge
        and the final clamp is what bounds the answer.
        """
        bottom_px: float = (roi.y + roi.height) * (self._rows - 1)
        below: float = bottom_px - self._horizon_px          # px below the horizon
        if below <= 0.0:
            # Feet at or above the horizon: not standing on this floor.
            return _MAX_DISTANCE
        distance: float = self._camera_height * self._focal / below
        return max(_MIN_DISTANCE, min(_MAX_DISTANCE, distance))

    def estimate_height(self, roi: Rect) -> float:
        """How tall the person is (m), from the box's top and bottom rows.

        A PURE PIXEL RATIO, which is the gift of the cylindrical frame: the rows below the
        horizon *are* the tangent of the depression, so

            height = camera_height * (bottom_px - top_px) / (bottom_px - horizon_px)

        The focal length, the field of view, the tilt and the distance all cancel, because the
        person and the camera stand on the same floor — the classic single-view horizon ratio.
        On rows linear in elevation it would have taken the distance and two arctangents.

        SCALE-FREE AND PARALLAX-FREE. One camera sees both the feet and the head, so unlike the
        azimuth this needs no ring correction, and two cameras at different distances from the
        same person agree in metres while their *pixel* box heights differ by tens of percent.
        That makes it the honest quantity to match observations on across a seam, where
        ``seam.max_height_diff`` compares frame fractions that genuinely disagree.

        IT READS REACH, NOT STATURE. The box top is the highest pixel, so raised arms read
        about 2.2 m where the same person reads 1.8 m with their arms down, at any distance.
        Here that is the useful number: overhead reach is what the tilt is chosen around.

        Accuracy is the distance estimate's in relative terms, since it is the same denominator:
        a pixel of box noise is a centimetre, while a degree of horizon error is 9 cm at 1.5 m
        and 35 cm at 7 m. Reads 0.0 when the feet sit at or above the horizon — nobody standing
        on this floor — and is capped at ``_MAX_HEIGHT``.
        """
        rows: int = self._rows - 1
        below: float = (roi.y + roi.height) * rows - self._horizon_px
        if below <= 0.0:
            return 0.0
        return min(_MAX_HEIGHT, self._camera_height * roi.height * rows / below)

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

    def set_window(self, window: FrameWindow, rows: int) -> None:
        """The delivered frame's row model: `rows` tall, horizon at `window.horizon_px`,
        `window.focal` px per unit of tangent."""
        self._window: FrameWindow = window
        self._rows: int = max(2, rows)
        self._horizon_px: float = window.horizon_px
        self._focal: float = max(1e-6, window.focal)
