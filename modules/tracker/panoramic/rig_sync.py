# Standard library imports
import math
from functools import lru_cache
from typing import Callable

# Third-party imports
import numpy as np

# Local application imports
from modules.oak import FrameWindow, degrees_per_pixel, delivered_height, frame_coverage, \
    frame_window, mode_size
from .projection import camera_bearing, focus_distance
from .rig import Rig
from .settings import RigSettings, TrackerSettings

# The reference person's overhead reach (m): 1.8 m tall, hands at 2.2 m with the arms up —
# CALIBRATION.md, *Tilt — derived from the build*. The height the `hands_*` read-outs are
# quoted at. A constant, not a setting: nothing tunes it, and the fields it feeds are read-only.
HANDS_HEIGHT: float = 2.2


@lru_cache(maxsize=8)
def _coverage(src: tuple[int, int], rows: int, fov: float, tilt: float, lens_fov: float,
              lens_centre: tuple[float, float]) -> np.ndarray:
    """`frame_coverage` for one frame configuration, computed once per process.

    It projects the whole output grid (≈0.1 s) and every input is an init field, so the running app
    pays for it once — but the tests build trackers by the dozen, mostly on the same few frames.
    Read-only, since every caller shares the one array.
    """
    coverage: np.ndarray = frame_coverage(src, (src[0], rows), src[0], fov, tilt,
                                          lens_fov=lens_fov, lens_centre=lens_centre)
    coverage.flags.writeable = False
    return coverage


def reach_radius(height: float, centre_bearing: float, camera_height: float, camera_radius: float,
                 limit: Callable[[float], float], far: float = 50.0) -> float:
    """The nearest radius (m) from the fixture, along the line `centre_bearing` off a camera's
    axis, at which a point `height` m above the floor is still inside that camera's picture.

    `height` 0 gives where the feet enter the frame, 2.2 m where raised hands do — the numbers the
    tilt is chosen by. `limit(camera_bearing)` is the picture's edge at that camera column, in
    degrees from eye level (its top for a point above the lens, its bottom for one below), NaN
    where the column has no picture. Per column, because the sensor's top edge falls toward the
    frame edges, and a person on the line is seen at the wider `camera_bearing`, not the line's
    own bearing — so the seam reach is searched rather than read off one column.

    Monotonic in the radius, so a bisection over `[camera_radius, far]` finds it; `inf` if not in
    frame even at `far`. On the axis it is `(height - camera_height) / tan(limit) + camera_radius`.
    """
    rise: float = height - camera_height

    def in_frame(radius: float) -> bool:
        distance: float = focus_distance(centre_bearing, camera_radius, radius)
        if distance <= 1e-9:
            return False
        angle: float = math.degrees(math.atan(rise / distance))
        edge: float = limit(camera_bearing(centre_bearing, camera_radius, distance))
        return angle <= edge if rise >= 0.0 else angle >= edge     # NaN edge: both False

    lo: float = max(0.0, camera_radius)
    hi: float = max(lo, far)
    if not in_frame(hi):
        return math.inf
    for _ in range(60):
        mid: float = (lo + hi) / 2.0
        if in_frame(mid):
            hi = mid
        else:
            lo = mid
    return hi


class RigSync:
    """Keeps the `Rig` in step with `RigSettings`, and publishes every read-only field the tracker
    derives, so the panel and the panorama read the numbers the tracker actually tracks with.

    The frame is derived once, here in the constructor, because every input to it is an init field.
    `apply()` covers the live ones — the camera radius, the lens height, the foot offset and the zone — and
    always pushes all of them, so no ordering between them can matter. Not thread-safe on its own:
    the tracker calls it on its own thread.
    """

    def __init__(self, config: TrackerSettings, rig: Rig) -> None:
        self._config: TrackerSettings = config
        self._rig: Rig = rig
        # Where the sensor's picture ends, per camera bearing: (top, bottom) in degrees from eye
        # level. The reach read-outs are measured against it.
        self._picture_edges: tuple[Callable[[float], float], Callable[[float], float]]
        self._set_frame()
        self.apply()

    def apply(self) -> None:
        """Push the live rig settings into the `Rig` and republish what they move."""
        c: TrackerSettings = self._config
        r: RigSettings = c.rig
        self._rig.set_camera_radius(r.camera_radius)
        self._rig.set_camera_height(r.camera_height)
        self._rig.set_foot_offset(c.foot_offset)
        self._rig.set_zone(r.zone_min_radius, r.zone_max_radius)
        # Published so the panorama draws its marks on the same cylinder the azimuth is corrected at.
        r.parallax_radius = self._rig.parallax_radius
        # In world azimuth, as the panorama's degree grid measures it; the local-angle band
        # `angle_in_overlap` tests stays inside the `Rig`.
        r.overlap = self._rig.overlap_azimuth
        self._publish_reach()

    def _set_frame(self) -> None:
        """The delivered frame's geometry, from the functions the camera's warp is built with: `fov`
        for the columns, `frame_window` for the rows (tangents of elevation, so a window rather than
        a `vfov`). Published as the frame's two edge angles, from which `projection.row_model`
        rebuilds the row form exactly. Mono and landscape."""
        c: TrackerSettings = self._config
        fov: float = c.fov
        self._rig.set_fov(fov)
        lens_centre: tuple[float, float] = (c.lens_centre_x, c.lens_centre_y)
        src: tuple[int, int] = mode_size(False, c.resolution)
        rows: int = delivered_height(False, c.resolution, fov, c.tilt, c.lens_fov, lens_centre)
        window = frame_window(src, (src[0], rows), src[0], fov, c.tilt, c.lens_fov, lens_centre)
        self._rig.set_window(window, rows)
        p: RigSettings = c.rig
        p.hfov = fov
        p.vfov = window.elevation_top - window.elevation_bottom
        p.tilt = c.tilt
        p.angle_bottom = window.elevation_bottom
        p.angle_top = window.elevation_top
        self._set_picture_edges(_coverage(src, rows, fov, c.tilt, c.lens_fov, lens_centre),
                                window, src[0], fov)

    def _set_picture_edges(self, coverage: np.ndarray, window: FrameWindow, width: int,
                           fov: float) -> None:
        """Per camera bearing, the angle where the sensor's picture ends: its top and its bottom.

        Not the frame's rows: the sensor fills less than the frame toward the sides (the black arch),
        so a reach judged against the rows would be too generous. NaN outside the field or on a
        column with no picture. Columns are taken as `coverage_summary` takes them.
        """
        dpp: float = degrees_per_pixel(fov, width)
        centre: float = (width - 1) / 2.0

        def edge(bearing: float, which: int) -> float:
            if dpp <= 0.0:
                return math.nan
            x: float = centre + bearing / dpp
            if x < -0.5 or x > width - 0.5:
                return math.nan
            row: int = int(coverage[int(round(min(max(x, 0.0), width - 1.0))), which])
            return window.elevation(row) if row >= 0 else math.nan

        self._picture_edges = (lambda bearing: edge(bearing, 0), lambda bearing: edge(bearing, 1))

    def _publish_reach(self) -> None:
        """How near the fixture a person can stand and still be in frame — the tilt's trade, live.

        Feet on the camera axis only: the frame is pinned at the sensor's lowest centre-column
        reach, so the bottom row is covered at every bearing. Raised hands on the axis and on the
        seam, the worse of the seam's two sides (a lens-centre offset makes them differ): the
        sensor's top edge falls toward the frame edges, and a seam is where people cross.
        """
        top, bottom = self._picture_edges
        r: RigSettings = self._config.rig
        camera_radius: float = max(0.0, r.camera_radius)
        seam: float = self._rig.target_fov / 2.0
        r.feet_from = reach_radius(0.0, 0.0, r.camera_height, camera_radius, bottom)
        r.hands_from = reach_radius(HANDS_HEIGHT, 0.0, r.camera_height, camera_radius, top)
        r.hands_seam = max(reach_radius(HANDS_HEIGHT, side, r.camera_height, camera_radius, top)
                           for side in (-seam, seam))
