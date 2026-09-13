import math

from modules.oak import FrameWindow, frame_window
from modules.utils import Rect

from .annotation import Annotation
from .projection import azimuth_to_camera_x, camera_azimuth, camera_local_to_azimuth, \
    centre_distance, wrap180


# Tallest reading `estimate_height` will report (m). Feet a pixel below the horizon divide by almost
# nothing, so the ratio needs a ceiling — above any real reach (~2.4 m with arms up), so it only ever
# catches a mangled box.
_MAX_HEIGHT: float = 3.0


def height_is_measured(height: float) -> bool:
    """Whether an `estimate_height` reading is a measurement rather than one of its two escapes.

    0.0 (feet at or above the horizon) and `_MAX_HEIGHT` (feet barely below it) both mean "no
    reading", not "this tall". A gate comparing two heights asks this first, so a person mid-jump is
    not dropped at the moment they are hardest to re-find.
    """
    return 0.0 < height < _MAX_HEIGHT


class Rig:
    """The installation as the tracker models it — the camera ring, the lens height, the tracked
    zone and the delivered frame — and what a camera's box means in it: world azimuth, distance and
    height (`annotate`), and where a column falls relative to the field edges, the overlap and the
    zone.

    `RigSync` keeps it in step with `RigSettings`. The delivered frame is **level** and
    **cylindrical** — a column is one azimuth at every height, a row one elevation (as its tangent)
    at every column — because the camera's warp makes it so (`modules/oak/camera/definitions.py`,
    `warp_mesh_points`), so there is no distortion correction here.
    """

    def __init__(self, cam_fov: float, target_fov: float) -> None:
        self.cam_fov: float = cam_fov
        self.target_fov: float = target_fov

        # Lens distance from the fixture axis (m); 0 disables the parallax correction.
        self._ring_radius: float = 0.0
        # How far below the feet the detector's box bottom sits, in frame heights (`_foot_px`).
        self._foot_offset: float = 0.0
        # Lens height above the floor (m) — the one measured constant the distance estimate needs.
        self._camera_height: float = 0.5
        # The tracked floor, as radii from the fixture axis. `set_zone` replaces these defaults.
        self._min_radius: float = 1.5
        self._max_radius: float = 3.5
        # Derived by `_update_overlap_band` and `_update_parallax_depth`; they must exist before
        # `set_zone` derives them.
        self.overlap_band: float = self.cam_fov - self.target_fov
        self.overlap_azimuth: float = self.cam_fov - self.target_fov
        self.parallax_radius: float = (self._min_radius + self._max_radius) / 2.0
        self.set_zone(self._min_radius, self._max_radius)

        # Until `set_window`, an untilted 1280 x 800 frame with the ideal lens.
        self.set_window(frame_window((1280, 800), (1280, 800), 1280, cam_fov, 0.0), 800)

    def annotate(self, roi: Rect, cam_id: int) -> Annotation:
        """Everything one box says about one person.

        The overlap flag is the *picture* — this column is inside the band a neighbour also sees —
        not a tunable; whether two observations are one person is `seam.link_angle`.
        """
        local_angle, world_angle, distance = self.calc_angle(roi, cam_id)
        return Annotation(local_angle, world_angle, self.angle_in_overlap(local_angle), distance,
                          self.estimate_height(roi))

    def calc_angle(self, roi: Rect, cam_id: int) -> tuple[float, float, float]:
        """Local angle, world azimuth and distance (m) for one box.

        **The world azimuth does not use the distance**: it goes through `camera_local_to_azimuth`
        at the fixed `parallax_radius` (why: `_update_parallax_depth`), the stitch's own chain, so
        the tracker and `azimuth_to_camera_x` are one exact inverse pair. Edge, overlap, dead-zone,
        hysteresis and re-acquisition tests all stay in the camera's own `local_angle`; only the
        world azimuth is comparable between cameras.
        """
        local_angle: float = self._calc_local_angle(roi)
        world_angle: float = self._local_to_azimuth(local_angle, cam_id)
        return local_angle, world_angle, self.estimate_distance(roi)

    def column_to_azimuth(self, cam_id: int, x: float) -> float:
        """World azimuth (degrees, [0, 360)) of a normalised column of camera `cam_id`.

        The chain `calc_angle` puts a box centre through, so a keypoint's column and its box
        centre's are directly comparable in world degrees.
        """
        return self._local_to_azimuth(x * self.cam_fov, cam_id)

    def _local_to_azimuth(self, local_angle: float, cam_id: int) -> float:
        return camera_local_to_azimuth(local_angle, cam_id, self.cam_fov, self.target_fov,
                                       self._ring_radius, self.parallax_radius)

    def _foot_px(self, roi: Rect) -> float:
        """The row (px) the feet are on: the box bottom less `foot_offset`.

        The one place the correction exists, shared by `estimate_distance` and `estimate_height`.
        The detector's box bottom sits below the feet by what measures as a fixed pad in pixels, so
        it is subtracted as a fraction of frame height rather than scaled. The ROI itself is never
        rewritten, so `height_filter` and the crop extractor still see the detector's box.
        """
        return (roi.y + roi.height - self._foot_offset) * (self._rows - 1)

    def estimate_distance(self, roi: Rect) -> float:
        """Distance from the camera (m), from where the feet meet the floor.

        The rows below the horizon are the tangent of the depression, so on a floor plane
        `distance = camera_height * focal / (foot_px - horizon_px)`. Nothing about the person enters
        it — raised arms or bending over change a box's height, not its feet. The tilt is inside the
        window (`set_window`, derived with the warp's own `frame_window`): the horizon is not the
        centre row, and none of this holds on footage that has not been through the warp.

        The frame's bottom row bounds it from below (the sensor's lowest reach on the centre column);
        anyone closer has their feet below the frame. A box may extend past the frame edge — the
        detector extrapolates a partly visible person — and the formula is continuous across it, but
        it is a guess there.

        Reads short until `foot_offset` is calibrated (CALIBRATION.md, *The tracker's height*).
        Unclamped: the far-edge filter (`beyond_zone`) and the panorama's `R` and foot tick need a
        reading past the zone to be past the zone. Feet at or above the horizon read `inf`.
        """
        below: float = self._foot_px(roi) - self._horizon_px  # px below the horizon
        if below <= 0.0:
            return math.inf
        return self._camera_height * self._focal / below

    def beyond_zone(self, local_angle: float, distance: float) -> bool:
        """Whether a person at this column and camera distance stands past the zone's far edge.

        Tested as a **radius from the fixture**, not a camera distance: a camera sits `ring_radius`
        out toward the person, so on its own axis at R 3.6 it reads 3.24 m, which a camera-distance
        test against 3.5 would accept. The radius is also what two cameras at a seam agree on.

        **Far edge only**: near the fixture the feet are often below the frame and the foot row is
        the detector's guess. No margin: brief errors are absorbed by the tracker's timeouts, steady
        ones are calibration. An uncalibrated `foot_offset` reads everyone nearer, so this fails open.
        """
        radius: float = centre_distance(local_angle - self.cam_fov / 2.0, distance, self._ring_radius)
        return radius > self._max_radius

    def estimate_height(self, roi: Rect) -> float:
        """How tall the person is (m), from the box's top and bottom rows.

        A pure pixel ratio on the cylindrical frame, the single-view horizon ratio:

            height = camera_height * (foot_px - top_px) / (foot_px - horizon_px)

        Focal, field, tilt and distance all cancel, and so does the ring: one camera sees both feet
        and head, so two cameras at different distances agree in metres. That is what
        ``seam.link_height`` compares, as a fraction of the larger reading.

        It reads **reach, not stature** — raised arms read ~2.2 m where arms down read 1.8 m — and it
        is the signal `foot_offset` is tuned by: with the detector's pad uncorrected it reads low and
        *falls* with distance; tune until it is flat (CALIBRATION.md, *The tracker's height*). 0.0
        when the feet are at or above the horizon; capped at ``_MAX_HEIGHT``.
        """
        rows: int = self._rows - 1
        foot_px: float = self._foot_px(roi)
        below: float = foot_px - self._horizon_px
        if below <= 0.0:
            return 0.0
        span: float = foot_px - roi.y * rows                 # head to corrected feet, px
        return min(_MAX_HEIGHT, max(0.0, self._camera_height * span / below))

    def _calc_local_angle(self, roi: Rect) -> float:
        """The bearing of the box centre within this camera's field. Exact: the frame is
        cylindrical."""
        normalized_x: float = roi.x + roi.width / 2.0
        return normalized_x * self.cam_fov

    def angle_in_overlap(self, local_angle: float) -> bool:
        """Is this column inside the band a neighbouring camera also sees — `overlap_band` in from
        either field edge?

        The precondition for a seam link, not a tunable. Deliberately not per person: the true band
        narrows with distance, but too wide costs nothing (`Seams.linked_world` finds no partner)
        while too narrow splits a person, so it is derived once at the zone's far edge, where it is
        widest (`_update_overlap_band`).
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

    def _update_parallax_depth(self) -> None:
        """The one depth the world azimuth is corrected at: the zone's **harmonic mean**.

        **Why a fixed depth and not each person's own.** The only per-person length is
        `estimate_distance`, off the box bottom, which reads short — and the two cameras at a seam
        err in opposite directions, doubling the disagreement. Seam disagreement for one person, in
        degrees:

            distance used           R 1.5   R 2.25   R 3.5   worst
            no correction at all    23.1     14.5     9.0     23.1
            per-person, 50% short   16.5     11.6     7.8     16.5
            fixed, R 2.1             6.1      1.4     6.3      6.3
            per-person, exact        0.0      0.0     0.0      0.0

        A real body's own seam disagreement is 5.9 / 2.3 / 0.85 at those radii, so the fixed depth
        already sits at the irreducible floor. A bias is calibratable; a detector box's per-person
        variance is not.

        **Why the harmonic mean.** The correction is linear in `1/d`, so its worst case over the
        zone is smallest at the midpoint of `1/d`: `2·min·max/(min+max)`, R 2.1 for R 1.5–3.5 (the
        true optimum, R 2.06, buys a third of a degree).

        **Two depths from one zone, on purpose.** The overlap flag (`_update_overlap_band`) takes the
        far edge, because a gate on whether a link is attempted must never under-report; this takes
        the middle, because a correction wants its worst case smallest.
        """
        self.parallax_radius = (2.0 * self._min_radius * self._max_radius
                                / max(1e-6, self._min_radius + self._max_radius))

    def _update_overlap_band(self) -> None:
        """The overlap threshold, in the two frames it is needed in.

        **`overlap_band`, a local angle, at the zone's far edge**, where a camera's coverage of the
        room is widest (see `_update_parallax_depth` for why the far edge). The neighbour's field
        begins `target_fov - half_span` off this camera's axis, with `half_span` the centre azimuth
        from the axis to the field edge; `azimuth_to_camera_x` turns that back into a local angle.

        **`overlap_azimuth`, the same threshold in azimuth at `parallax_radius`**, because that is
        the depth the panorama draws marks at: the line then sits exactly where a mark's field changes
        width. It is not the geometric overlap at that depth — the flag is generous,
        and the line inherits that.

        At `camera_radius = 0` both are `cam_fov - target_fov`. The local band is clamped to
        `[0, cam_fov/2]`, so it cannot become always-true on a ring of more, narrower sectors.
        """
        bare: float = max(0.0, self.cam_fov - self.target_fov)
        if self._ring_radius <= 0.0 or self._max_radius <= 0.0:
            self.overlap_band = bare
            self.overlap_azimuth = bare
            return

        # The local threshold, at the zone's far edge.
        axis: float = camera_azimuth(0, self.target_fov)
        edge: float = camera_local_to_azimuth(self.cam_fov, 0, self.cam_fov, self.target_fov,
                                              self._ring_radius, self._max_radius)
        half_span: float = wrap180(edge - axis)
        x: float | None = azimuth_to_camera_x(axis + self.target_fov - half_span, 0, self.cam_fov,
                                              self.target_fov, self._ring_radius, self._max_radius)
        band: float = self.cam_fov - x * self.cam_fov if x is not None else 0.0
        self.overlap_band = min(max(0.0, band), self.cam_fov / 2.0)

        # The same threshold in azimuth, at the depth the marks are drawn on, measured from the
        # seam this camera's far edge sits on (`target_fov`, one sector along from its own axis 0).
        # No threshold, no line: a zero band must stay zero here, not become the distance from the
        # seam to an unthresholded field edge.
        if self.overlap_band <= 0.0:
            self.overlap_azimuth = 0.0
            return
        inner: float = camera_local_to_azimuth(self.cam_fov - self.overlap_band, 0, self.cam_fov,
                                               self.target_fov, self._ring_radius,
                                               self.parallax_radius)
        self.overlap_azimuth = max(0.0, 2.0 * wrap180(self.target_fov - inner))

    # SET
    def set_fov(self, cam_fov: float) -> None:
        self.cam_fov = cam_fov
        self._update_overlap_band()

    def set_camera_radius(self, camera_radius: float) -> None:
        """How far each lens sits from the fixture axis (m). Re-derives the zone, which depends on
        the ring too."""
        self._ring_radius = max(0.0, camera_radius)
        self.set_zone(self._min_radius, self._max_radius)

    def set_camera_height(self, camera_height: float) -> None:
        self._camera_height = camera_height

    def set_foot_offset(self, foot_offset: float) -> None:
        """How far below the feet the detector's box bottom sits, in frame heights. See `_foot_px`."""
        self._foot_offset = max(0.0, foot_offset)

    def set_zone(self, min_radius: float, max_radius: float) -> None:
        """The tracked floor, as two radii from the fixture axis. Derives the parallax depth, the
        overlap band, and the far edge `beyond_zone` tests."""
        self._min_radius = max(0.0, min_radius)
        self._max_radius = max(self._min_radius, max_radius)
        self._update_parallax_depth()
        self._update_overlap_band()

    def set_window(self, window: FrameWindow, rows: int) -> None:
        """The delivered frame's row model: `rows` tall, horizon at `window.horizon_px`,
        `window.focal` px per unit of tangent."""
        self._window: FrameWindow = window
        self._rows: int = max(2, rows)
        self._horizon_px: float = window.horizon_px
        self._focal: float = max(1e-6, window.focal)
