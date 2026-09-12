import math

from modules.oak import FrameWindow, frame_window
from modules.utils import Rect

from .panorama_map import azimuth_to_camera_x, camera_azimuth, camera_local_to_azimuth, wrap180


# Tallest reading `estimate_height` will report (m). A box whose feet land a pixel below the
# horizon divides by almost nothing, so the ratio needs a ceiling. Set above anything a person
# can measure — a tall one with both arms up reaches ~2.4 — so the cap only ever catches a
# mangled box, never a real reading it might otherwise flatten.
_MAX_HEIGHT: float = 3.0


def height_is_measured(height: float) -> bool:
    """Whether an `estimate_height` reading is a measurement rather than one of its two escapes.

    `estimate_height` returns 0.0 when the feet sit at or above the horizon — nobody standing on
    this floor — and `_MAX_HEIGHT` when they barely clear it, where the ratio divides by almost
    nothing. Both are "no reading", not "this tall". Any gate that compares two heights has to
    ask this first: a person mid-jump has their feet off the floor and reads high or capped, and
    gating on that would drop them at the very moment they are hardest to re-find.
    """
    return 0.0 < height < _MAX_HEIGHT


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
        # How far local 0 sits before this camera's own sector starts — the offset in
        # `_calc_world_angle`, and half of what the field has spare. Depth-free and exact: it is a
        # mounting fact, not a width. Do not confuse it with `overlap_band` below, which is the
        # *width* two cameras share and is nearly four times larger here.
        self.fov_overlap: float = (self.cam_fov - self.target_fov) / 2.0

        # Parallax: cameras sit on a ring of this radius (m), not at the shared
        # centre the world-angle model assumes. 0 disables the correction.
        self._ring_radius: float = 0.0
        # Lens height above the floor (m) — the one measured constant the distance estimate needs.
        self._camera_height: float = 0.5
        # The tracked floor, as radii from the rig centre, and the camera distances they bound.
        # `set_zone` derives the second pair from the first; these are its defaults.
        self._min_radius: float = 1.5
        self._max_radius: float = 3.5
        self._min_distance: float = 1.5
        self._max_distance: float = 3.5
        # The part of this camera's field a neighbour also sees, in local angle and in world
        # azimuth. `_update_overlap_band` is the one place either is derived; it needs the ring,
        # so it runs after it exists rather than beside `fov_overlap`.
        self.overlap_band: float = self.cam_fov - self.target_fov
        self.overlap_world: float = self.cam_fov - self.target_fov
        self.set_zone(self._min_radius * 2.0, self._max_radius * 2.0)

        # The frame's rows: where the horizon is (px) and how many px a unit of tangent spans.
        # Until `set_window`, an untilted 1280 x 800 frame with the ideal lens.
        self.set_window(frame_window((1280, 800), (1280, 800), 1280, cam_fov, 0.0), 800)

    def get_angles_and_overlap(self, roi: Rect, cam_id: int) -> tuple[float, float, bool, float, float]:
        """Everything one box says about one person, in the order `Annotation` holds it:
        local angle, world angle, overlap flag, distance (m) and height (m).

        The overlap flag is the *picture*: this column is inside the band a neighbouring camera
        also sees. Nothing tunable widens it — how close two observations must be to be one
        person is `seam.link_angle`, a separate question asked in world degrees.
        """
        local_angle, world_angle, distance = self.calc_angle(roi, cam_id)
        overlap: bool = self.angle_in_overlap(local_angle)
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

        **THE CLAMP IS A GUARD, NOT A FILTER.** Nothing is rejected for falling outside it; only
        the reading is pinned. It matters because the denominator can go to almost nothing (feet a
        pixel below the horizon reads as infinitely far) or the box can be extrapolated far below
        the frame (reads as zero), and this distance feeds the parallax correction, which rotates
        the world azimuth. Unclamped, one mangled box swings a person's bearing arbitrarily; the
        bounds mean it can only move within the band people are actually in. They are derived from
        the tracked zone by `set_zone`, so they follow the ring instead of being hand-computed for
        one.
        """
        bottom_px: float = (roi.y + roi.height) * (self._rows - 1)
        below: float = bottom_px - self._horizon_px          # px below the horizon
        if below <= 0.0:
            # Feet at or above the horizon: not standing on this floor.
            return self._max_distance
        distance: float = self._camera_height * self._focal / below
        return max(self._min_distance, min(self._max_distance, distance))

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
        That is what ``seam.link_height`` compares, as a percentage of the larger of the two:
        scale-free, so the lens height cancels out of it as well.

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

    def angle_in_overlap(self, local_angle: float) -> bool:
        """Is this column inside the band a neighbouring camera also sees?

        A property of the lens, the mount and the zone — `overlap_band` in from each field edge,
        not `fov_overlap` — and not a tuning knob. It says *a second opinion exists here*, which
        is the precondition for linking; whether two opinions are the same person is
        `seam.link_angle`.

        **DELIBERATELY NOT PER-PERSON**, though the world angle beside it is. The true shared band
        narrows with distance, so an exact answer would need this person's own `estimate_distance`
        — and that is the one quantity still unverified (see CALIBRATION.md, *Open*). The cost is
        asymmetric: too wide costs nothing, because `_find_world_candidate` then runs and finds no
        partner within `link_angle`; too narrow splits one person into two worlds at a seam. So
        the band is derived once, at the zone's FAR edge, where it is widest — the most generous
        depth-free bound that still never under-reports anywhere people are tracked.
        """
        return local_angle <= self.overlap_band or local_angle >= self.cam_fov - self.overlap_band

    def angle_in_edge(self, local_angle: float, degrees: float) -> bool:
        """Is this column within `degrees` of either end of the camera's own field?"""
        return local_angle <= degrees or local_angle >= self.cam_fov - degrees

    def angle_from_edge(self, local_angle: float) -> float:
        return min(local_angle, self.cam_fov - local_angle)

    @staticmethod
    def angle_diff(a: float, b: float) -> float:
        diff: float = abs(a - b)
        if diff > 180.0:
            diff = 360.0 - diff
        return diff

    def _update_overlap_band(self) -> None:
        """Derive both overlap widths, at the tracked zone's far edge. The one place either moves.

        A camera `r` out from the centre covers less of the room than its bare field suggests, and
        how much less depends on the depth. Take the depth where the coverage is widest — the
        zone's outer circle — and the geometry closes:

            half_span      = the azimuth, at the centre, from a camera's axis to its field edge
            overlap_world  = 2 * half_span - target_fov          (what two neighbours share)

        and the neighbour's field begins `target_fov - half_span` off this camera's axis, which
        `azimuth_to_camera_x` turns back into a local angle — the threshold `angle_in_overlap`
        needs. The two differ (28.3° local against 26.4° of azimuth on this rig) because the
        local-to-azimuth map compresses toward a field edge.

        At `camera_diameter = 0` the depth question disappears with the parallax correction, and
        both collapse to `cam_fov - target_fov`. The local band is clamped to `[0, cam_fov/2]`:
        zero is right below the diameter where the sectors stop meeting at all, and the ceiling
        keeps `angle_in_overlap` from becoming always-true on a ring of more, narrower sectors.
        """
        bare: float = max(0.0, self.cam_fov - self.target_fov)
        diameter: float = self._max_radius * 2.0
        if self._ring_radius <= 0.0 or diameter <= 0.0:
            self.overlap_band = bare
            self.overlap_world = bare
            return

        axis: float = camera_azimuth(0, self.target_fov)
        edge: float = camera_local_to_azimuth(self.cam_fov, 0, self.cam_fov, self.target_fov,
                                              self._ring_radius, diameter)
        half_span: float = wrap180(edge - axis)
        self.overlap_world = max(0.0, 2.0 * half_span - self.target_fov)

        x: float | None = azimuth_to_camera_x(axis + self.target_fov - half_span, 0, self.cam_fov,
                                              self.target_fov, self._ring_radius, diameter)
        band: float = self.cam_fov - x * self.cam_fov if x is not None else 0.0
        self.overlap_band = min(max(0.0, band), self.cam_fov / 2.0)

    # SET
    def set_fov(self, cam_fov: float) -> None:
        self.cam_fov = cam_fov
        self.fov_overlap = (self.cam_fov - self.target_fov) / 2.0
        self._update_overlap_band()

    def set_camera_diameter(self, camera_diameter: float) -> None:
        """The ring the lenses sit on, as a **diameter** — the setting's own unit. Radii live only
        in here, since that is what the parallax triangle takes."""
        self._ring_radius = max(0.0, camera_diameter) / 2.0
        self.set_zone(self._min_radius * 2.0, self._max_radius * 2.0)

    def set_camera_height(self, camera_height: float) -> None:
        self._camera_height = camera_height

    def set_zone(self, min_diameter: float, max_diameter: float) -> None:
        """The tracked floor, as the two **diameters** the settings carry.

        Two things follow. The overlap band, at the far edge (`_update_overlap_band`). And the
        distance clamp, as camera distances: the on-axis extremes, since a camera is pushed
        `ring_radius` toward the circle it faces and away from the one behind it — so the nearest
        anyone in the zone can be is `min_radius - ring_radius`, and the furthest
        `max_radius + ring_radius`. Ø 3 to Ø 7 on a Ø 0.72 ring gives 1.14 m to 3.86 m.
        """
        self._min_radius = max(0.0, min_diameter) / 2.0
        self._max_radius = max(self._min_radius, max_diameter / 2.0)
        self._min_distance = max(0.01, self._min_radius - self._ring_radius)
        self._max_distance = max(self._min_distance, self._max_radius + self._ring_radius)
        self._update_overlap_band()

    def set_window(self, window: FrameWindow, rows: int) -> None:
        """The delivered frame's row model: `rows` tall, horizon at `window.horizon_px`,
        `window.focal` px per unit of tangent."""
        self._window: FrameWindow = window
        self._rows: int = max(2, rows)
        self._horizon_px: float = window.horizon_px
        self._focal: float = max(1e-6, window.focal)
