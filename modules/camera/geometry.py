"""The frame geometry every camera shares: sensor modes, the delivered frame, the lens, the warp
mesh, the coverage, and the mount readout. Pure numpy and math; no device package is imported, so
all of it is testable without hardware and every backend, the tracker and the render derive the same
numbers from the same settings."""

import math
from dataclasses import dataclass
from enum import IntEnum, auto
from typing import cast

import numpy as np

from modules.utils import Rect

import logging
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Frame geometry — the one source for every camera dimension
# ---------------------------------------------------------------------------
# One knob: the sensor mode (`resolution`), a setting so it sits beside every preset value that
# depends on it. Every width, height, aspect ratio and buffer size in the pipeline and in the
# render derives from it and the mount. Nothing downstream may restate a frame dimension as a
# literal — that is how the two silently drift.
#
# The delivered frame's height is derived, not set: the warp's rows are tangents of elevation
# (`FrameWindow`), which spend rows at the top, so a tilted camera's full reach needs more rows
# than the sensor has (1152 at P800 and tilt 16; 960 at P720 and tilt 15). `delivered_height`
# computes it and the open log prints it. The warp's output size is free; only its alignment is not.

# `dai.node.Warp` needs both output dimensions divisible by this.
WARP_ALIGNMENT: int = 16


class CameraResolution(IntEnum):
    """The sensor mode, as one label spanning both sensors — the `resolution` setting.

    Named `CameraResolution` rather than `Resolution` because `modules/pose` has a `resolution`
    of its own and the apps import both.

    It is a *setting* and not a constant on purpose. Everything frame-relative in the preset is
    tuned against a particular frame, and a constant in code with the values that depend on it in
    a preset is two sources that can disagree. It is also what lets a recording be played back
    honestly: a clip shot at 720 rows runs through an 800-row pipeline with `vfov`, the distance
    estimate and the panorama's geometry all silently wrong, and a playback preset saying `P720`
    fixes every one of them at once.

    NOT derived from it, and to be re-tuned on the rig alongside it, because they are set by eye
    against the frame rather than computed (P800 -> P720 scales them by 800/720 = 1.111):
      track.height_filter                   a fraction of frame height, so it scales
      track.foot_offset                     likewise: the detector's pad measures in pixels
    (`track.seam.*` is NOT in that list either: it is in degrees and fractions of a height.)
    (The vertical field is NOT in that list — see `frame_fov` below, which derives it. Neither is
    the render's panorama row, which derives its own aspect.)
    """
    P720  = 0       # 1280 x 720   both sensors
    P800  = auto()  # 1280 x 800   both sensors; the OV9282 W's full readout
    P1080 = auto()  # 1920 x 1072  colour only


# The delivered size of each mode, already trimmed to the warp's alignment: 1080 is not
# divisible by 16, so the 1080p preview is cropped to 1072 rows.
RESOLUTION_SIZES: dict[CameraResolution, tuple[int, int]] = {
    CameraResolution.P720:  (1280,  720),
    CameraResolution.P800:  (1280,  800),
    CameraResolution.P1080: (1920, 1072),
}

# Which modes each sensor actually has. Checked against depthai 2.30: both MonoCamera and
# ColorCamera expose THE_720_P and THE_800_P, which is what lets one label span them. Mono also
# has THE_400_P, THE_480_P and THE_1200_P — the last is for AR0234-class sensors, not the
# OV9282 W in use — and colour has 4 K upward; none are offered until something needs them.
# A backend maps the label to its own sensor mode enum (`modules/camera/oak`, `mono_mode`, `color_mode`).
MONO_RESOLUTIONS: frozenset[CameraResolution] = frozenset({CameraResolution.P720, CameraResolution.P800})
COLOR_RESOLUTIONS: frozenset[CameraResolution] = frozenset({CameraResolution.P720, CameraResolution.P800,
                                                             CameraResolution.P1080})


# What a mono camera falls back to when asked for a mode it does not have.
_FALLBACK_RESOLUTION: CameraResolution = CameraResolution.P800

# Pairs already reported, so the warning says its piece once. `frame_size` runs per frame on the
# simulator's size check, and a misconfigured preset must not turn that into a log flood.
_warned_resolutions: set[tuple[bool, CameraResolution]] = set()


def resolve_resolution(color: bool, resolution: CameraResolution) -> CameraResolution:
    """The mode this sensor will actually run, which is not always the one asked for.

    `P1080` is colour-only, and a preset can ask a mono camera for it. Rather than raising in the
    middle of pipeline construction, fall back and say so once — the frame is still coherent,
    just not the requested size."""
    available = COLOR_RESOLUTIONS if color else MONO_RESOLUTIONS
    if resolution in available:
        return resolution
    if (color, resolution) not in _warned_resolutions:
        _warned_resolutions.add((color, resolution))
        logger.warning(
            "%s is not available on the %s sensor — falling back to %s",
            resolution.name, 'colour' if color else 'mono', _FALLBACK_RESOLUTION.name,
        )
    return _FALLBACK_RESOLUTION


# Heights already reported as not aligned, so the warning says its piece once per value.
_warned_heights: set[int] = set()


def aligned_height(height: int) -> int:
    """`height` rounded to the warp's alignment, warning once when that changes it."""
    aligned: int = int(round(height / WARP_ALIGNMENT)) * WARP_ALIGNMENT
    if aligned != height and height not in _warned_heights:
        _warned_heights.add(height)
        logger.warning("frame height %d is not a multiple of %d — using %d", height, WARP_ALIGNMENT, aligned)
    return max(WARP_ALIGNMENT, aligned)


def _delivered(mode: tuple[int, int], square: bool, height: int) -> tuple[int, int]:
    """The delivered frame for a sensor mode: `square` crops it to the mode's height x height;
    otherwise the mode's width by `height` rows (0 = the mode's own)."""
    width, mode_height = mode
    if square:
        return (mode_height, mode_height)
    return (width, aligned_height(height) if height > 0 else mode_height)


def mono_frame_size(resolution: CameraResolution, square: bool = False, height: int = 0) -> tuple[int, int]:
    """The mono frame as it leaves the pipeline. `square` crops it to height x height;
    `height` (px, 0 = the sensor mode's rows) is the warp's output height otherwise."""
    return _delivered(RESOLUTION_SIZES[resolve_resolution(False, resolution)], square, height)


def color_frame_size(resolution: CameraResolution, square: bool = False, height: int = 0) -> tuple[int, int]:
    """The colour frame as it leaves the pipeline. `square` crops it to height x height;
    `height` (px, 0 = the sensor mode's rows) is the warp's output height otherwise."""
    return _delivered(RESOLUTION_SIZES[resolve_resolution(True, resolution)], square, height)


def frame_size(color: bool, resolution: CameraResolution, square: bool = False,
               height: int = 0) -> tuple[int, int]:
    """The frame size a camera configuration produces, whichever path it takes. Anything that
    needs the camera's aspect ratio — the render's layout, a texture allocation — asks here
    instead of restating it. `height` is the delivered height (`delivered_height`)."""
    return (color_frame_size(resolution, square, height) if color
            else mono_frame_size(resolution, square, height))


def delivered_height(color: bool, resolution: CameraResolution, fov_h: float, tilt: float,
                     lens_fov: float = 0.0, lens_center: tuple[float, float] = (0.0, 0.0)) -> int:
    """The delivered frame's rows: the sensor's full reach at this tilt (`full_frame_height`).
    Every consumer of the frame's shape (the camera, the tracker, the render) calls it with the
    same shared settings, so they agree without the number being stored anywhere."""
    src: tuple[int, int] = mode_size(color, resolution)
    return full_frame_height(src, src[0], fov_h, tilt, lens_fov, lens_center)


def full_frame_height(src_size: tuple[int, int], mode_width: int, fov_h: float, tilt: float,
                      lens_fov: float = 0.0,
                      lens_center: tuple[float, float] = (0.0, 0.0)) -> int:
    """The frame height (aligned) at which the centre column carries every sensor row: the
    window pinned at the sensor's bottom reach needs this many tangent rows to reach its top.
    Logged at open so the number to put in the preset is never worked out by hand."""
    focal: float = output_focal(fov_h, mode_width)
    lens_focal, _, cy = source_lens(src_size, mode_width, fov_h, lens_fov, lens_center)
    if focal <= 0.0 or lens_focal <= 0.0:
        return src_size[1]
    bottom: float = math.radians(tilt) - (src_size[1] - 1 - cy) / lens_focal
    top: float = math.radians(tilt) + cy / lens_focal
    rows: float = focal * (math.tan(top) - math.tan(bottom)) + 1.0
    return int(math.ceil(rows / WARP_ALIGNMENT)) * WARP_ALIGNMENT


def mode_size(color: bool, resolution: CameraResolution) -> tuple[int, int]:
    """The frame size BEFORE any square crop — the size the sensor mode actually produces.
    This is the width the `fov` setting is quoted against, so it is what `degrees_per_pixel`
    must be given."""
    return frame_size(color, resolution, square=False)


# ---------------------------------------------------------------------------
#  Lens geometry — the lens is three shared numbers; the frame is a contract
# ---------------------------------------------------------------------------
# Sensor reference. EVERY figure below is for the sensor's FULL readout, so it is true only in
# the mode that reads every line. Pairing a field with the wrong mode is how `vfov` went stale
# at 720 rows.
#
#   device        sensor                   native        DFOV  HFOV  VFOV
#   OAK-D Pro W   OV9282 mono pair         1280 x 800     150   127  79.5
#   OAK-D Pro W   IMX378 colour (stock)    4056 x 3040    120    95  72
#   OAK-D Pro W   OV9782 colour (upgrade)  1280 x 800     150   127  79.5
#   OAK-1 W       IMX378 colour            4056 x 3040    120    95  72
#   OAK-1 W       OV9782 colour            1280 x 800     150   127  79.5
#
#   https://docs.luxonis.com/hardware/products/OAK-D%20Pro%20W
#   https://docs.luxonis.com/hardware/products/OAK-1%20W
#
# `get_device_list(verbose=True)` logs the sensor behind each socket, so the hardware answers
# the "which variant is this?" question itself rather than it being inferred.
#
# TWO DIFFERENT THINGS USED TO BE ONE NUMBER. `fov` is the field the DELIVERED FRAME spans —
# a contract the tracker builds on (`cam_fov`, the overlaps, the azimuth of a column). The
# LENS is what the sensor actually sees, and it is not `fov`: read off the four White Space
# units' factory calibrations (mono CAM_B, 1280 x 800), every lens is equidistant to 0.5 %
# out to 75 deg off axis, but at 565-575 px/rad rather than the 577.5 that `fov = 127`
# implies, with the optical centre 5-24 px left and ~10 px below the frame centre:
#
#   unit             focal px/rad   cx      cy      field across 1280 px
#   ...F124D9D600    575.4          615.2   409.9   127.8
#   ...110AD3D200    567.2          634.3   407.6   128.8
#   ...31DDD2D200    566.7          632.6   411.7   128.7
#   ...1136D1D200    565.0          634.5   410.3   129.6
#
# Modelling the lens as `fov` read bearings ~1 deg short toward the seams and put the horizon
# 10 px (1 deg) too high — a metre of distance error at 5 m. So the lens is its own three
# settings, shared by all cameras of an installation: `lens_fov` (the field the lens spans
# across the full sensor width; 0 means "same as fov", the old model), `lens_center_x` and
# `lens_center_y` (optical centre offset from the frame centre, px, at the full sensor mode —
# invariant under the 720-row centre crop). One shared lens rather than one per unit was a
# deliberate trade: the mean leaves the outlier unit 1.4 deg off at its azimuth zero, and
# `lens_deviation` reports each unit's residual at open so that stays visible.
#
# `fov` must not exceed the lens's field, or the edge columns read outside the sensor and go
# black; `build_warp_mesh` warns.


def degrees_per_pixel(fov_h: float, mode_width: int) -> float:
    """Angular size of one output column (degrees) — the frame's horizontal scale.

    ``mode_width`` must be the width of the UN-cropped frame the sensor mode produces,
    because ``fov_h`` is quoted for that full width. A later crop — the square crop, or the
    1080 -> 1072 trim — removes pixels without changing this scale, which is exactly why the
    scale rather than a field angle is the thing worth deriving from.
    """
    if mode_width <= 0:
        return 0.0
    return fov_h / float(mode_width)


def output_focal(fov_h: float, mode_width: int) -> float:
    """The delivered frame's focal length (px per radian): one column per `dpp`, and the
    same scale for the rows at the horizon — see `FrameWindow`."""
    dpp: float = degrees_per_pixel(fov_h, mode_width)
    return 1.0 / math.radians(dpp) if dpp > 0.0 else 0.0


def source_lens(src_size: tuple[int, int], mode_width: int, fov_h: float,
                lens_fov: float = 0.0,
                lens_center: tuple[float, float] = (0.0, 0.0)) -> tuple[float, float, float]:
    """The lens as the warp needs it: (focal px/rad, cx, cy) in source-pixel coordinates.

    Equidistant: a ray `phi` radians off the optical axis lands `focal * phi` px from
    `(cx, cy)`. ``lens_fov`` is the field across the full ``mode_width``; 0 falls back to
    ``fov_h``, which is the model that was in use before the calibrations were read.
    """
    src_w, src_h = src_size
    field: float = lens_fov if lens_fov > 0.0 else fov_h
    focal: float = mode_width / math.radians(field) if field > 0.0 and mode_width > 0 else 0.0
    # Pixel-index convention: a frame of width W spans 0..W-1, so its centre sits at (W-1)/2,
    # not W/2. The half pixel matters — it is what keeps the centre exactly on axis.
    return focal, (src_w - 1) / 2.0 + lens_center[0], (src_h - 1) / 2.0 + lens_center[1]


# Mesh resolution for the warp. See `warp_mesh_points` for why 2 is not enough and why 32 is
# where this stops mattering.
WARP_MESH: int = 32


@dataclass(frozen=True)
class FrameWindow:
    """The rows of a delivered frame, as elevations: the vertical half of the frame contract.

    ROWS ARE CYLINDRICAL: `horizon_px - row = focal * tan(elevation)`, with the same `focal`
    as the columns (`output_focal`), so the frame is locally a level pinhole camera panned to
    each column's bearing. That is what keeps a body's proportions true at any height in the
    frame, and it is chosen for the pose model, which was trained on pinhole photographs;
    equirectangular rows (linear in elevation) shrink a metre at 40 deg elevation by 41 % and
    at 55 deg by 67 % relative to eye level.

    THE WINDOW IS PINNED AT THE BOTTOM. The last row is the sensor's lowest reach on the centre
    column, `tilt - reach_below_axis`, and the rows run upward from there for as many as the
    frame has. Pinned there because the feet are what the tilt rule pins (the floor-plane
    distance reads off the feet), and because the tangent spends rows at the TOP: 800 rows at
    tilt 16 reach 43 deg, 1152 rows reach the sensor's own 57 deg. The horizon may lie outside
    the frame; nothing here treats that as special.

    `elevation_bottom`, `elevation_top` in degrees; `horizon_px` the (float, un-flipped) row of
    elevation 0; `focal` in px per radian. `flip_v` mirrors the delivered rows, so the delivered
    row of `horizon_px` is then `out_h - 1 - horizon_px`.
    """
    elevation_bottom: float
    elevation_top: float
    horizon_px: float
    focal: float

    def elevation(self, row: float) -> float:
        """Elevation (degrees) of an un-flipped output row."""
        if self.focal <= 0.0:
            return 0.0
        return math.degrees(math.atan((self.horizon_px - row) / self.focal))

    def row(self, elevation: float) -> float:
        """Un-flipped output row (float) of an elevation (degrees)."""
        return self.horizon_px - self.focal * math.tan(math.radians(elevation))


def frame_window(src_size: tuple[int, int], out_size: tuple[int, int], mode_width: int,
                 fov_h: float, tilt: float, lens_fov: float = 0.0,
                 lens_center: tuple[float, float] = (0.0, 0.0)) -> FrameWindow:
    """The `FrameWindow` a configuration delivers. Pure; the same call the warp makes."""
    out_h: int = out_size[1]
    src_h: int = src_size[1]
    focal: float = output_focal(fov_h, mode_width)
    lens_focal, _, cy = source_lens(src_size, mode_width, fov_h, lens_fov, lens_center)
    if focal <= 0.0 or lens_focal <= 0.0:
        return FrameWindow(0.0, 0.0, (out_h - 1) / 2.0, 0.0)
    reach_below: float = math.degrees((src_h - 1 - cy) / lens_focal)
    bottom: float = tilt - reach_below
    horizon: float = (out_h - 1) + focal * math.tan(math.radians(bottom))
    top: float = math.degrees(math.atan(horizon / focal))
    return FrameWindow(bottom, top, horizon, focal)


def horizon_row(window: FrameWindow, out_h: int) -> float:
    """Normalised row (0 = top) of the horizon in a delivered frame; may fall outside [0, 1].

    Every consumer that turns a row into an elevation needs the window, not this number alone:
    rows are tangents, so `(row - horizon_row) * something` is never an angle. Kept for callers
    that only need to draw the horizon.
    """
    return window.horizon_px / (out_h - 1) if out_h > 1 else 0.5


def frame_fov(fov_h: float, mode_width: int, out_size: tuple[int, int],
              window: FrameWindow) -> tuple[float, float]:
    """The horizontal and vertical field (degrees) a delivered frame actually covers.

    The horizontal is `dpp * width`, so it survives the square crop and every resolution
    change. The vertical is the window's span, which is NOT `dpp * height`: the rows are
    tangents and the window is pinned at the bottom, so it depends on the tilt and the lens.
    """
    dpp: float = degrees_per_pixel(fov_h, mode_width)
    return dpp * out_size[0], window.elevation_top - window.elevation_bottom


def lens_field(focal: float, width: int) -> float:
    """The field (degrees) an equidistant lens of `focal` px/rad spans across `width` px."""
    return math.degrees(width / focal) if focal > 0.0 else float('nan')


def lens_deviation(focal: float, cx: float, cy: float, src_size: tuple[int, int],
                   mode_width: int, fov_h: float, lens_fov: float = 0.0,
                   lens_center: tuple[float, float] = (0.0, 0.0)) -> float:
    """The largest bearing error (degrees) a unit with this calibration suffers under the
    shared lens: the larger of its optical-centre offset and the focal mismatch at the frame
    edge. A bound, for the open log and the panel, not a correction."""
    shared_focal, shared_cx, shared_cy = source_lens(src_size, mode_width, fov_h, lens_fov, lens_center)
    if shared_focal <= 0.0 or focal <= 0.0:
        return float('nan')
    center: float = math.hypot(cx - shared_cx, cy - shared_cy) / shared_focal
    edge: float = (mode_width / 2.0) * abs(1.0 / shared_focal - 1.0 / focal)
    return math.degrees(max(center, edge))


def _project_grid(
    src_size: tuple[int, int],
    out_size: tuple[int, int],
    mode_width: int,
    fov_h: float,
    tilt: float,
    flip_h: bool,
    flip_v: bool,
    xs: np.ndarray,
    ys: np.ndarray,
    lens_fov: float,
    lens_center: tuple[float, float],
) -> np.ndarray:
    """The source pixel each output pixel of the grid `ys x xs` reads: shape (len(ys), len(xs), 2).

    The one place the frame contract meets the lens. Output pixel -> world ray (column = bearing
    about the vertical, row = tangent of elevation, `FrameWindow`), rotate the ray by the tilt
    into the camera's frame, project it through the equidistant lens (`source_lens`).
    """
    src_w, src_h = src_size
    out_w, out_h = out_size
    lens_focal, cx, cy = source_lens(src_size, mode_width, fov_h, lens_fov, lens_center)
    window: FrameWindow = frame_window(src_size, out_size, mode_width, fov_h, tilt, lens_fov, lens_center)
    X, Y = np.meshgrid(np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64))
    if window.focal <= 0.0 or lens_focal <= 0.0:
        # No field to speak of: hand the frame through, the square crop cut from the centre.
        return np.stack([X + (src_w - out_w) / 2.0, Y + (src_h - out_h) / 2.0], axis=-1)

    ox: np.ndarray = (out_w - 1.0) - X if flip_h else X
    oy: np.ndarray = (out_h - 1.0) - Y if flip_v else Y
    # Output pixel -> world ray. Image y is down, so d1 is the DOWNWARD component.
    bearing: np.ndarray = (ox - (out_w - 1.0) / 2.0) * math.radians(degrees_per_pixel(fov_h, mode_width))
    elev: np.ndarray = np.arctan((window.horizon_px - oy) / window.focal)
    cos_e: np.ndarray = np.cos(elev)
    d0: np.ndarray = cos_e * np.sin(bearing)
    d1: np.ndarray = -np.sin(elev)
    d2: np.ndarray = cos_e * np.cos(bearing)
    # Undo the tilt: rotate about the horizontal axis by -tilt (the camera is aimed UP by tilt).
    t: float = math.radians(-tilt)
    cos_t, sin_t = math.cos(t), math.sin(t)
    d1, d2 = cos_t * d1 - sin_t * d2, sin_t * d1 + cos_t * d2
    # Ray -> source pixel, equidistant: radius is the polar angle off the axis times the focal.
    r: np.ndarray = np.arccos(np.clip(d2, -1.0, 1.0)) * lens_focal
    psi: np.ndarray = np.arctan2(d1, d0)
    return np.stack([cx + r * np.cos(psi), cy + r * np.sin(psi)], axis=-1)


def warp_mesh_points(
    src_size: tuple[int, int],
    out_size: tuple[int, int],
    mode_width: int,
    fov_h: float,
    tilt: float,
    flip_h: bool = False,
    flip_v: bool = False,
    mesh_w: int = WARP_MESH,
    mesh_h: int = WARP_MESH,
    lens_fov: float = 0.0,
    lens_center: tuple[float, float] = (0.0, 0.0),
) -> list[tuple[float, float]]:
    """Warp mesh that makes a column one azimuth, a row one tangent of elevation, and undoes
    the camera's up-tilt: the source pixel each output grid point reads, row-major.

    THE OUTPUT IS CYLINDRICAL, NOT THE LENS'S OWN PROJECTION. These lenses are equidistant:
    distance from the optical centre is proportional to the ANGLE off the axis. That projection
    has a property the tracker cannot live with — a column is one azimuth only on the
    horizontal centre line. Off it, a vertical pole bows toward the centre, so a standing
    person's box centre reads short of the true bearing: 5 deg at 45 deg bearing 1.35 m out,
    8 deg at the seam. No 1-D correction on x can fix that; it is a property of the projection.
    So the output pixel (x, y) is read as (bearing, elevation): the column linear in bearing,
    the row linear in the TANGENT of elevation (`FrameWindow`), which is a level pinhole camera
    panned to that column — the picture the pose model was trained on. Every column is one
    azimuth at every height; every row is one elevation at every column.

    WHY THIS IS NOT A HOMOGRAPHY. A pinhole lens uses the tangent on both axes, and that is
    exactly what makes a tilted pinhole view a trapezoid with straight edges, which
    `cv2.getPerspectiveTransform` reproduces. On an equidistant lens nothing cancels: tilting
    bends straight lines into curves. At 15 deg on a 127 deg frame the centre of a row moves
    151 px while its ends move 74 px, a 77 px bow. So the mesh is built by unprojecting each
    output pixel to a ray, rotating the ray, and reprojecting into the equidistant source.

    WHY THE MESH IS 32 x 32. The Warp node interpolates linearly between mesh points, so
    2 columns can only express a straight source line per row — which is the very family the
    homography already spanned, and the bow above is what the correction IS. This is a
    threshold, not a spectrum: 2 is unusable, ~16 is already sub-pixel, and 32 x 32 differs
    from 64 x 64 by under 0.3 px even at 30 deg. Error grows linearly with frame width and
    falls with the square of the mesh count, so 32 still holds to 0.23 px on a 4056 px frame.

    `src_size` is the frame the sensor delivers and `out_size` the warp's output, which is
    smaller only for the square crop. `mode_width` is the un-cropped width `fov_h` and
    `lens_fov` are quoted against. Positive `tilt` means the camera is aimed UP. The lens is
    `source_lens`'s three numbers; the window `frame_window`'s.

    WHERE THE SENSOR DID NOT LOOK, THE OUTPUT IS BLACK. The sensor is a rectangle in the lens's
    projection, and that is not a rectangle in bearing/elevation, so no rectangular window fills
    without cutting. With the window pinned at the sensor's bottom reach, the centre column is
    covered top to bottom (for as far as the rows go) and the TOP goes black toward the sides in
    an arch, because rotating a sideways ray about the horizontal axis lifts it by less than
    the tilt: at tilt 16 on 800 rows the arch is ~0 px at the centre column and grows toward the
    edge columns. `frame_coverage` says exactly where, per column, and it is what downstream
    should ask instead of assuming the frame's top row. Mesh points that fall outside the
    source are left where they land; the Warp node paints them black.

    The fixed point is the LENS CENTRE, not the frame centre: the output's centre column at the
    row of elevation `tilt` reads the source pixel `(cx, cy)`. At `tilt == 0` the result is NOT
    the identity — it is the equidistant -> cylindrical reprojection.
    """
    out_w, out_h = out_size
    grid: np.ndarray = _project_grid(
        src_size, out_size, mode_width, fov_h, tilt, flip_h, flip_v,
        np.linspace(0.0, out_w - 1.0, mesh_w), np.linspace(0.0, out_h - 1.0, mesh_h),
        lens_fov, lens_center)
    return [(float(x), float(y)) for x, y in grid.reshape(-1, 2)]


def frame_coverage(
    src_size: tuple[int, int],
    out_size: tuple[int, int],
    mode_width: int,
    fov_h: float,
    tilt: float,
    flip_h: bool = False,
    flip_v: bool = False,
    lens_fov: float = 0.0,
    lens_center: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """Per output column, the first and last row the sensor actually covers: shape (out_w, 2),
    int, `(-1, -1)` for a column with no picture at all.

    The whole output grid projected through the same path as the mesh (one numpy pass, a few
    tens of ms — for open time, not per frame) and tested against the source rectangle: what
    the Warp node will paint, to within its bilinear interpolation of the 32 x 32 mesh. Rows
    are in DELIVERED order, flips included. This is the function that tells the tracker and the
    panorama where the black arch is, so a raised arm near a seam is judged against what the
    camera could see rather than the frame's top row.
    """
    src_w, src_h = src_size
    out_w, out_h = out_size
    pts: np.ndarray = _project_grid(
        src_size, out_size, mode_width, fov_h, tilt, flip_h, flip_v,
        np.arange(out_w), np.arange(out_h), lens_fov, lens_center)
    # A pixel's footprint is half a pixel either side of its index; the edge rows and columns
    # count as covered up to that, which also keeps float noise on an exact edge from losing them.
    inside: np.ndarray = np.asarray((pts[..., 0] >= -0.5) & (pts[..., 0] <= src_w - 0.5)
                                    & (pts[..., 1] >= -0.5) & (pts[..., 1] <= src_h - 0.5))
    # First and last covered row: the black is an arch at the top or the bottom, so a column's
    # covered rows are one run. The one exception is a column at the very edge of the field
    # losing a pixel or two NEAR THE HORIZON where the lens's reach falls short of `fov`; that
    # is not a coverage shape but a preset fact, and `_report_lens_reach` (pipeline.py) logs
    # it at open with its size.
    covered: np.ndarray = np.asarray(inside.any(axis=0))
    first: np.ndarray = inside.argmax(axis=0)
    last: np.ndarray = out_h - 1 - inside[::-1, :].argmax(axis=0)
    return np.stack([np.where(covered, first, -1), np.where(covered, last, -1)], axis=1).astype(int)


def coverage_summary(coverage: np.ndarray, out_size: tuple[int, int], mode_width: int,
                     fov_h: float, flip_h: bool = False) -> str:
    """One line for the open log: the rows covered on the centre column, at the seams (45 deg
    off axis, the worse side) and at the edge columns (idem)."""
    out_w, out_h = out_size
    dpp: float = degrees_per_pixel(fov_h, mode_width)
    center: float = (out_w - 1) / 2.0

    def column(bearing: float) -> int:
        x: float = center +(bearing / dpp if dpp > 0.0 else 0.0)
        x = (out_w - 1) - x if flip_h else x
        return int(round(min(max(x, 0.0), out_w - 1.0)))

    def span(columns: list[int]) -> str:
        rows = coverage[columns]
        if (rows[:, 0] < 0).any():
            return 'none'
        return f'{int(rows[:, 0].max())}-{int(rows[:, 1].min())}'

    return (f'rows covered of {out_h}: centre {span([column(0.0)])}, '
            f'seams {span([column(-45.0), column(45.0)])}, '
            f'edges {span([0, out_w - 1])}')


# ---------------------------------------------------------------------------
#  Mount readout — what the camera can say about its own orientation
# ---------------------------------------------------------------------------
# Every camera constant in this installation is either measured with a tape (the camera radius,
# the lens height) or taken from a datasheet (`fov`). Two are neither: `tilt` depends on a hand
# adjustment on a tripod head, and ROLL is not modelled anywhere at all — `warp_mesh_points`
# takes a tilt and assumes the camera is level about its optical axis. A rolled camera tilts the
# horizon, which in the stitched panorama looks exactly like a wrong `tilt`, so the image cannot
# tell the two apart. The IMU can.
#
# This is a setup aid and nothing in the show may read it: a few hertz, and every value is
# allowed to be missing.


def orientation_from_gravity(gx: float, gy: float, gz: float) -> tuple[float, float]:
    """Tilt and roll (degrees) of a camera, from the gravity vector in ITS OWN frame.

    The camera frame is x right, y down, z forward (depthai's convention), so a level camera sees gravity at
    ``(0, 1, 0)`` and one aimed straight up sees it at ``(0, 0, -1)``. Tilt is positive aimed
    up, matching the `tilt` setting; roll is positive rolled toward +x.

    The decomposition is **separable**: pure tilt returns roll 0 and pure roll returns tilt 0,
    at any magnitude. That is the property worth having — it is what lets a mount error be
    attributed to one axis or the other instead of arriving as one combined number, which is
    exactly the ambiguity the panorama already suffers from.

    Returns NaN for both if the vector has no length (a dead sensor reads all zeros, and a
    zero-length vector has no orientation to report).
    """
    magnitude: float = math.sqrt(gx * gx + gy * gy + gz * gz)
    if magnitude < 1e-6:
        return (float('nan'), float('nan'))
    tilt: float = math.degrees(math.atan2(-gz, math.hypot(gx, gy)))
    roll: float = math.degrees(math.atan2(gx, gy))
    return (tilt, roll)


# ---------------------------------------------------------------------------
#  The board's IMU: a vector from the IMU's frame into the camera's
# ---------------------------------------------------------------------------
# `orientation_from_gravity` turns the gravity vector into tilt and roll; these get the vector
# there. A backend reads the IMU-to-camera matrix off its device's calibration and knows the
# turn its board mounts the IMU at (`oak.definitions.IMU_BOARD_ROLL`, measured on the OAK-D).

def imu_to_camera(vector: tuple[float, float, float],
                  extrinsics: list[list[float]] | None) -> tuple[float, float, float]:
    """Rotate a vector from the IMU's frame into a camera's.

    ``extrinsics`` is the 4x4 from ``CalibrationHandler.getImuToCameraExtrinsics``; only its
    rotation block is used, since a direction has no position. A missing or malformed matrix
    falls through as the identity: on a board where the two frames happen to agree that is
    right, and where they do not it is at least not silently wrong in an invented direction.
    """
    if not extrinsics or len(extrinsics) < 3:
        return vector
    try:
        return cast(tuple[float, float, float], tuple(
            sum(extrinsics[row][col] * vector[col] for col in range(3))
            for row in range(3)
        ))
    except (IndexError, TypeError):
        return vector


def unroll_imu_frame(vector: tuple[float, float, float], board_roll: float) -> tuple[float, float, float]:
    """Undo the IMU's mounting rotation about the optical axis, `board_roll` degrees.

    A rotation about z, so it corrects the roll and cannot disturb the tilt. Under this matrix a
    rotation by `angle` maps a roll of phi to phi - angle, so the angle to apply IS the offset
    being removed, not its negation.
    """
    angle: float = math.radians(board_roll)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    x, y, z = vector
    return (x * cos_a - y * sin_a, x * sin_a + y * cos_a, z)


# ---------------------------------------------------------------------------
#  Tilt: the picture of a level camera
# ---------------------------------------------------------------------------
# A tilt is a rotation about the camera, so where a pixel lands does not depend on how far away it
# is: one homography straightens a frame and the depth aligned to it. It holds for a rectilinear
# frame only, so an undistorted one. The third coordinate it gives a pixel is that pixel's depth
# scale: a distance along the tilted axis times it is the distance along the level one.

MIN_CORNER_SCALE: float = 0.1       # a frame corner past the horizon has no place in the level picture


def tilt_homography(intrinsics: list[list[float]], size: tuple[int, int], tilt: float) -> np.ndarray:
    """From the uv of a frame `size` to the uv of the same frame from a level camera, for a camera
    `tilt` degrees up; `intrinsics` is the frame's 3x3 lens in pixels. The picture is scaled and
    moved so all of it stays in the frame, and tilt 0 is the identity. The result is not rescaled:
    its third row is the depth scale of a pixel (`H[2] @ (u, v, 1)`)."""
    width, height = size
    lens: np.ndarray = np.array(intrinsics, dtype=np.float64)
    lens /= lens[2, 2]
    cos_t, sin_t = math.cos(math.radians(tilt)), math.sin(math.radians(tilt))
    rotation: np.ndarray = np.array([[1.0, 0.0, 0.0], [0.0, cos_t, -sin_t], [0.0, sin_t, cos_t]])
    level: np.ndarray = lens @ rotation @ np.linalg.inv(lens)

    corners: np.ndarray = level @ np.array([[0.0, 0.0, 1.0], [width, 0.0, 1.0], [width, height, 1.0], [0.0, height, 1.0]]).T
    x, y = corners[:2] / np.maximum(corners[2], MIN_CORNER_SCALE)
    span_x, span_y = float(x.max() - x.min()), float(y.max() - y.min())
    scale: float = min(width / span_x, height / span_y)
    fit: np.ndarray = np.array([[scale, 0.0, (width - scale * span_x) / 2.0 - scale * float(x.min())],
                                [0.0, scale, (height - scale * span_y) / 2.0 - scale * float(y.min())],
                                [0.0, 0.0, 1.0]])
    to_uv: np.ndarray = np.diag([1.0 / width, 1.0 / height, 1.0])
    return to_uv @ fit @ level @ np.diag([float(width), float(height), 1.0])


def homography_points(homography: np.ndarray, points: np.ndarray) -> np.ndarray:
    """uv points (N, 2) through a homography."""
    mapped: np.ndarray = homography @ np.column_stack((points, np.ones(len(points)))).T
    return (mapped[:2] / mapped[2]).T


def corrected_roi(roi: Rect, homography: np.ndarray) -> Rect:
    """A normalised box straightened with the frame: the box round its four moved corners, in the frame."""
    x0, y0, x1, y1 = roi.x, roi.y, roi.x + roi.width, roi.y + roi.height
    corners: np.ndarray = homography_points(homography, np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]))
    left, top = np.clip(corners.min(axis=0), 0.0, 1.0)
    right, bottom = np.clip(corners.max(axis=0), 0.0, 1.0)
    return Rect(float(left), float(top), float(right - left), float(bottom - top))


def corrected_keypoints(keypoints: np.ndarray | None, homography: np.ndarray) -> np.ndarray | None:
    """Keypoints (N, 3) straightened with the frame; the scores stay."""
    if keypoints is None:
        return None
    corrected: np.ndarray = keypoints.copy()
    corrected[:, :2] = homography_points(homography, keypoints[:, :2])
    return corrected


def mirrored_intrinsics(intrinsics: list[list[float]], width: int) -> list[list[float]]:
    """A lens as it is for the frame flipped left to right: a recording stores mirrored frames."""
    mirrored: list[list[float]] = [list(row) for row in intrinsics]
    mirrored[0][2] = width - mirrored[0][2]
    return mirrored
