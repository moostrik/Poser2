# https://blobconverter.luxonis.com/

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from enum import Enum, IntEnum, auto
from typing import Callable, TypeAlias
from depthai import (Tracklet, TrackerType, ImgDetection, Rect, Point3f, Device, OpenVINO,
                     SpatialLocationCalculatorAlgorithm, MonoCameraProperties, ColorCameraProperties)

import logging
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Frame geometry — the one source for every camera dimension
# ---------------------------------------------------------------------------
# Two knobs: the sensor mode (`resolution`) and the delivered frame's height (`frame_height`).
# Every width, height, aspect ratio and buffer size in the pipeline and in the render derives
# from them. Nothing downstream may restate a frame dimension as a literal — that is how the
# two silently drift.
#
# Both are SETTINGS, not constants here. `resolution` used to be a constant, which put it in a
# different place from every preset value that depends on it; they could disagree and nothing
# noticed. `frame_height` exists because the warp's rows are tangents of elevation
# (`FrameWindow`), which spend rows at the top: a tilted camera's full reach needs more rows
# than the sensor has (1152 at P800 and tilt 16; 960 at P720 and tilt 15 — `full_frame_height`
# computes it and the open log prints it), and 0 keeps the sensor's own row count. The warp's
# output size is free; only its alignment is not.

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
      camera.tracker.min_height             a fraction of frame height, so it scales
      pose.distance_extractor.near_y/far_y  positions in the frame, so they scale and shift
    (`camera.tracker.seam.*` is NOT in that list either: it is in degrees and percent.)
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
MONO_MODES: dict[CameraResolution, MonoCameraProperties.SensorResolution] = {
    CameraResolution.P720: MonoCameraProperties.SensorResolution.THE_720_P,
    CameraResolution.P800: MonoCameraProperties.SensorResolution.THE_800_P,
}

COLOR_MODES: dict[CameraResolution, ColorCameraProperties.SensorResolution] = {
    CameraResolution.P720:  ColorCameraProperties.SensorResolution.THE_720_P,
    CameraResolution.P800:  ColorCameraProperties.SensorResolution.THE_800_P,
    CameraResolution.P1080: ColorCameraProperties.SensorResolution.THE_1080_P,
}

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
    available = COLOR_MODES if color else MONO_MODES
    if resolution in available:
        return resolution
    if (color, resolution) not in _warned_resolutions:
        _warned_resolutions.add((color, resolution))
        logger.warning(
            "%s is not available on the %s sensor — falling back to %s",
            resolution.name, 'colour' if color else 'mono', _FALLBACK_RESOLUTION.name,
        )
    return _FALLBACK_RESOLUTION


def mono_mode(resolution: CameraResolution) -> MonoCameraProperties.SensorResolution:
    """The depthai mono sensor mode for this label."""
    return MONO_MODES[resolve_resolution(False, resolution)]


def color_mode(resolution: CameraResolution) -> ColorCameraProperties.SensorResolution:
    """The depthai colour sensor mode for this label."""
    return COLOR_MODES[resolve_resolution(True, resolution)]


# Heights already reported as not aligned, so the warning says its piece once per value.
_warned_heights: set[int] = set()


def aligned_height(height: int) -> int:
    """`height` rounded to the warp's alignment, warning once when that changes it."""
    aligned: int = int(round(height / WARP_ALIGNMENT)) * WARP_ALIGNMENT
    if aligned != height and height not in _warned_heights:
        _warned_heights.add(height)
        logger.warning("frame_height %d is not a multiple of %d — using %d", height, WARP_ALIGNMENT, aligned)
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
    instead of restating it. `height` is the `frame_height` setting."""
    return (color_frame_size(resolution, square, height) if color
            else mono_frame_size(resolution, square, height))


def delivered_height(color: bool, resolution: CameraResolution, fov_h: float, tilt: float,
                     lens_fov: float = 0.0, lens_centre: tuple[float, float] = (0.0, 0.0),
                     frame_height: int = 0) -> int:
    """The `frame_height` setting resolved: the explicit value (aligned) when it is non-zero,
    otherwise the sensor's full reach at this tilt (`full_frame_height`).

    The one place 0 is given its meaning. The app fills the shared root field with this at
    startup so every consumer sees the same number, and the tracker calls it too, so it is
    right whether or not the app has done that yet.
    """
    if frame_height > 0:
        return aligned_height(frame_height)
    src: tuple[int, int] = mode_size(color, resolution)
    return full_frame_height(src, src[0], fov_h, tilt, lens_fov, lens_centre)


def full_frame_height(src_size: tuple[int, int], mode_width: int, fov_h: float, tilt: float,
                      lens_fov: float = 0.0,
                      lens_centre: tuple[float, float] = (0.0, 0.0)) -> int:
    """The `frame_height` (aligned) at which the centre column carries every sensor row: the
    window pinned at the sensor's bottom reach needs this many tangent rows to reach its top.
    Logged at open so the number to put in the preset is never worked out by hand."""
    focal: float = output_focal(fov_h, mode_width)
    lens_focal, _, cy = source_lens(src_size, mode_width, fov_h, lens_fov, lens_centre)
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
# across the full sensor width; 0 means "same as fov", the old model), `lens_centre_x` and
# `lens_centre_y` (optical centre offset from the frame centre, px, at the full sensor mode —
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
                lens_centre: tuple[float, float] = (0.0, 0.0)) -> tuple[float, float, float]:
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
    return focal, (src_w - 1) / 2.0 + lens_centre[0], (src_h - 1) / 2.0 + lens_centre[1]


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
                 lens_centre: tuple[float, float] = (0.0, 0.0)) -> FrameWindow:
    """The `FrameWindow` a configuration delivers. Pure; the same call the warp makes."""
    out_h: int = out_size[1]
    src_h: int = src_size[1]
    focal: float = output_focal(fov_h, mode_width)
    lens_focal, _, cy = source_lens(src_size, mode_width, fov_h, lens_fov, lens_centre)
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
                   lens_centre: tuple[float, float] = (0.0, 0.0)) -> float:
    """The largest bearing error (degrees) a unit with this calibration suffers under the
    shared lens: the larger of its optical-centre offset and the focal mismatch at the frame
    edge. A bound, for the open log and the panel, not a correction."""
    shared_focal, shared_cx, shared_cy = source_lens(src_size, mode_width, fov_h, lens_fov, lens_centre)
    if shared_focal <= 0.0 or focal <= 0.0:
        return float('nan')
    centre: float = math.hypot(cx - shared_cx, cy - shared_cy) / shared_focal
    edge: float = (mode_width / 2.0) * abs(1.0 / shared_focal - 1.0 / focal)
    return math.degrees(max(centre, edge))


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
    lens_centre: tuple[float, float],
) -> np.ndarray:
    """The source pixel each output pixel of the grid `ys x xs` reads: shape (len(ys), len(xs), 2).

    The one place the frame contract meets the lens. Output pixel -> world ray (column = bearing
    about the vertical, row = tangent of elevation, `FrameWindow`), rotate the ray by the tilt
    into the camera's frame, project it through the equidistant lens (`source_lens`).
    """
    src_w, src_h = src_size
    out_w, out_h = out_size
    lens_focal, cx, cy = source_lens(src_size, mode_width, fov_h, lens_fov, lens_centre)
    window: FrameWindow = frame_window(src_size, out_size, mode_width, fov_h, tilt, lens_fov, lens_centre)
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
    lens_centre: tuple[float, float] = (0.0, 0.0),
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
        lens_fov, lens_centre)
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
    lens_centre: tuple[float, float] = (0.0, 0.0),
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
        np.arange(out_w), np.arange(out_h), lens_fov, lens_centre)
    # A pixel's footprint is half a pixel either side of its index; the edge rows and columns
    # count as covered up to that, which also keeps float noise on an exact edge from losing them.
    inside: np.ndarray = np.asarray((pts[..., 0] >= -0.5) & (pts[..., 0] <= src_w - 0.5)
                                    & (pts[..., 1] >= -0.5) & (pts[..., 1] <= src_h - 0.5))
    # First and last covered row: the black is an arch at the top or the bottom, so a column's
    # covered rows are one run. The one exception is a column at the very edge of the field
    # losing a pixel or two NEAR THE HORIZON where the lens's reach falls short of `fov`; that
    # is not a coverage shape but a preset fact, and `_report_lens_reach` (pipeline.py) logs
    # it at open with its size.
    covered: np.ndarray = inside.any(axis=0)
    first: np.ndarray = inside.argmax(axis=0)
    last: np.ndarray = out_h - 1 - inside[::-1, :].argmax(axis=0)
    return np.stack([np.where(covered, first, -1), np.where(covered, last, -1)], axis=1).astype(int)


def coverage_summary(coverage: np.ndarray, out_size: tuple[int, int], mode_width: int,
                     fov_h: float, flip_h: bool = False) -> str:
    """One line for the open log: the rows covered on the centre column, at the seams (45 deg
    off axis, the worse side) and at the edge columns (idem)."""
    out_w, out_h = out_size
    dpp: float = degrees_per_pixel(fov_h, mode_width)
    centre: float = (out_w - 1) / 2.0

    def column(bearing: float) -> int:
        x: float = centre + (bearing / dpp if dpp > 0.0 else 0.0)
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


# The wide blobs are built on tools.luxonis.com from the stock ultralytics yolov8n weights,
# OpenVINO 2022.1, at 512 x 448 for the tall cylindrical frame (1280 x 1152 at P800 is 1.111;
# 512 x 448 is 1.143; the P720 frame at 1280 x 960 is 1.333, a 17 % mismatch that a 512 x 384
# blob would close). Same pixel count as the 640 x 352 they replace (229 k vs 225 k), so the
# same inference cost — confirmed on the rig, `tracker_fps` at the input rate on all four
# cameras. The 640 x 352 files stay in data/models for reference; nothing selects them.
YOLOV8_WIDE_5S: str = "yolov8n_coco_512x448_5S.blob"
YOLOV8_WIDE_6S: str = "yolov8n_coco_512x448_6S.blob"
YOLOV8_WIDE_7S: str = "yolov8n_coco_512x448_7S.blob"
YOLOV8_SQUARE_5S: str = "yolov8n_coco_416x416_5S.blob"
YOLOV8_SQUARE_6S: str = "yolov8n_coco_416x416_6S.blob"
YOLOV8_SQUARE_7S: str = "yolov8n_coco_416x416_7S.blob"
YOLO_CONFIDENCE_THRESHOLD: float = 0.5
YOLO_OVERLAP_THRESHOLD: float = 0.5


def detector_input_size(blob_path: str | Path) -> tuple[int, int]:
    """(width, height) the detector blob was compiled for, read from the blob itself.

    Not a camera dimension: the frame is resized into this whatever the sensor mode is, with
    `setKeepAspectRatio(False)` (see the Yolo setups in pipeline.py), so the frame is STRETCHED
    into the blob, never cropped and never letterboxed. Two consequences: nothing is lost at the
    frame edges, so nobody goes undetected there; and detections come back in normalized
    coordinates of the detector input, which a pure stretch preserves, so they map 1:1 onto the
    full frame. An aspect mismatch between frame and blob costs detection *quality*, never
    *geometry* — which is why the tracker's azimuths are right whatever blob is in use.

    Read off the blob rather than kept as a constant beside the file name, because the two used
    to be separate declarations that nothing checked against each other. depthai stores the
    input as [W, H, C, N].
    """
    blob = OpenVINO.Blob(Path(blob_path))
    dims = next(iter(blob.networkInputs.values())).dims
    return int(dims[0]), int(dims[1])

TRACKER_PERSON_LABEL: int = 0
TRACKER_TYPE: TrackerType = TrackerType.ZERO_TERM_IMAGELESS
# ZERO_TERM_COLOR_HISTOGRAM higher accuracy (but can drift when losing object)
# ZERO_TERM_IMAGELESS slightly faster

DEPTH_TRACKER_LOCATION: SpatialLocationCalculatorAlgorithm = SpatialLocationCalculatorAlgorithm.MIN
DEPTH_TRACKER_BOX_SCALE: float = 1.0
DEPTH_TRACKER_MIN_DEPTH: int = 500
DEPTH_TRACKER_MAX_DEPTH: int = 10000

class FrameType(Enum):
    NONE_ = 0
    VIDEO = 1
    LEFT_ = 2
    RIGHT = 3
    DEPTH = 4

FRAME_TYPE_NAMES: list[str] = [e.name for e in FrameType]

FRAME_TYPE_LABEL_DICT: dict[FrameType, str] = {
    FrameType.NONE_: 'N',
    FrameType.VIDEO: 'C',
    FrameType.LEFT_: 'L',
    FrameType.RIGHT: 'R',
    FrameType.DEPTH: 'S'
}

EXPOSURE_RANGE:     tuple[int, int] = (1000, 33000)
ISO_RANGE:          tuple[int, int] = ( 100, 1600 )
BALANCE_RANGE:      tuple[int, int] = (1000, 12000)
CONTRAST_RANGE:     tuple[int, int] = ( -10, 10   )
BRIGHTNESS_RANGE:   tuple[int, int] = ( -10, 10   )
LUMA_DENOISE_RANGE: tuple[int, int] = (   0, 4    )
SATURATION_RANGE:   tuple[int, int] = ( -10, 10   )
SHARPNESS_RANGE:    tuple[int, int] = (   0, 4    )

STEREO_DEPTH_RANGE: tuple[int, int] = ( 500, 15000)
STEREO_BRIGHTNESS_RANGE: tuple[int, int] = (   0, 255  )

class StereoMedianFilterType(Enum):
    OFF = 0
    KERNEL_3x3 = 1
    KERNEL_5x5 = 2
    KERNEL_7x7 = 3

STEREO_FILTER_NAMES: list[str] = [e.name for e in StereoMedianFilterType]

class CoderType(Enum):
    CPU =   0
    GPU =   1
    iGPU =  2

class CoderFormat(Enum):
    H264 = '.mp4'
    H265 = '.hevc'

FrameCallback: TypeAlias = Callable[[int, FrameType, np.ndarray], None]
SyncCallback: TypeAlias = Callable[[int, dict[FrameType, np.ndarray], float], None]
DetectionCallback: TypeAlias = Callable[[int, ImgDetection], None]
TrackerCallback: TypeAlias = Callable[[int, list[Tracklet]], None]
FPSCallback: TypeAlias = Callable[[int, float], None]

class Input(IntEnum):
    COLOR_CONTROL = auto()
    MONO_CONTROL = auto()
    STEREO_CONTROL = auto()
    VIDEO_FRAME_IN = auto()
    LEFT_FRAME_IN = auto()
    RIGHT_FRAME_IN = auto()

class Output(IntEnum):
    VIDEO_FRAME_OUT = auto()
    LEFT_FRAME_OUT = auto()
    RIGHT_FRAME_OUT = auto()
    STEREO_FRAME_OUT = auto()
    SYNC_FRAMES_OUT = auto()
    TRACKLETS_OUT = auto()
    IMU_OUT = auto()

def get_device_list(verbose: bool = False) -> list[str]:
    device_list: list[str] = []
    if verbose:
        logger.info('-- CAMERAS --------------------------------------------------')
    for device in Device.getAllAvailableDevices():
        device_list.append(device.getMxId())
        if verbose:
            logger.info(f"Camera: {device.getMxId()} {device.state}")
    if verbose:
        logger.info('-------------------------------------------------------------')
    return device_list


def log_connected_sensors(device: Device, device_id: str = '') -> None:
    """Log the sensor behind each socket of an already-open device.

    Which sensor a device carries decides its field of view, and the OAK-D Pro W and OAK-1 W
    both ship in two variants whose lenses differ by 32 degrees horizontally (see the sensor
    reference above). Reading it off the hardware beats inferring it from which resolutions a
    preset happens to request.
    """
    try:
        for feature in device.getConnectedCameraFeatures():
            logger.info(f'{device_id} sensor {feature.socket.name}: {feature.sensorName} '
                        f'{feature.width}x{feature.height}')
    except Exception as exc:                        # never let diagnostics break an open
        logger.debug(f'{device_id} could not read sensor features: {exc}')


# ---------------------------------------------------------------------------
#  Mount readout — what the camera can say about its own orientation
# ---------------------------------------------------------------------------
# Every camera constant in this installation is either measured with a tape (the ring radius,
# the lens height) or taken from a datasheet (`fov`). Two are neither: `tilt` depends on a hand
# adjustment on a tripod head, and ROLL is not modelled anywhere at all — `warp_mesh_points`
# takes a tilt and assumes the camera is level about its optical axis. A rolled camera tilts the
# horizon, which in the stitched panorama looks exactly like a wrong `tilt`, so the image cannot
# tell the two apart. The IMU can.
#
# This is a setup aid and nothing in the show may read it: a few hertz, and every value is
# allowed to be missing.

IMU_RATE_HZ: int = 5

# How hard each new sample pulls the running average. The camera is bolted to a tripod, so a
# single reading is almost entirely noise and there is no motion to track.
IMU_SMOOTHING: float = 0.1

# The IMU does not share the camera's axes: it sits a quarter turn about the optical axis, and
# `getImuToCameraExtrinsics` does not encode that rotation.
#
# MEASURED, NOT ASSUMED. On the rig all four cameras reported a roll of -90 degrees while their
# tilt read correctly, and that combination has exactly one cause: a rotation about the optical
# axis (z) leaves `gz` untouched and `hypot(gx, gy)` invariant, so tilt survives it intact while
# roll is displaced by a constant. Any other misalignment would have corrupted the tilt too.
#
# It is a property of the board, not of an installation, so it is a constant here rather than a
# setting — four separately adjusted tripods cannot agree on a quarter turn by coincidence.
#
# THIS CONSTANT CLAIMS THE QUARTER TURN AND NOTHING MORE. Calibrating against a spirit level left
# a residual of about -0.85 degrees shared by three of four units, and it is tempting to fold that
# in and call the constant -90.85. It is not folded in: -90 is a board *layout*, exact and
# evidenced, whereas -0.85 is the mean of three samples with no evidence a different set of boards
# shares it. Everything below a degree is per-unit and belongs in `CameraReadings.roll_offset`,
# where it is a measurement of one specific camera rather than an extrapolation from three. The
# fourth unit on this rig needs -2.25 while sitting level — a sensor fault, probably from a fall —
# which no shared constant could ever have absorbed.
IMU_BOARD_ROLL: float = -90.0


def orientation_from_gravity(gx: float, gy: float, gz: float) -> tuple[float, float]:
    """Tilt and roll (degrees) of a camera, from the gravity vector in ITS OWN frame.

    depthai's camera frame is x right, y down, z forward, so a level camera sees gravity at
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


def imu_to_camera(vector: tuple[float, float, float],
                  extrinsics: list[list[float]] | None) -> tuple[float, float, float]:
    """Rotate a vector from the IMU's frame into a camera's.

    ``extrinsics`` is the 4x4 from ``CalibrationHandler.getImuToCameraExtrinsics``; only its
    rotation block is used, since a direction has no position. A missing or malformed matrix
    falls through as the identity — on a board where the two frames happen to agree that is
    right, and where they do not it is at least not silently wrong in an invented direction.
    """
    if not extrinsics or len(extrinsics) < 3:
        return vector
    try:
        return tuple(                                       # type: ignore[return-value]
            sum(extrinsics[row][col] * vector[col] for col in range(3))
            for row in range(3)
        )
    except (IndexError, TypeError):
        return vector


def unroll_imu_frame(vector: tuple[float, float, float],
                     board_roll: float = IMU_BOARD_ROLL) -> tuple[float, float, float]:
    """Undo the IMU's mounting rotation about the optical axis.

    A rotation about z, so it corrects the roll and cannot disturb the tilt — which is the reason
    the tilt readout was already trustworthy before this existed. See ``IMU_BOARD_ROLL``.
    """
    # Under this matrix a rotation by `angle` maps a roll of phi to phi - angle, so the angle to
    # apply IS the offset being removed, not its negation.
    angle: float = math.radians(board_roll)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    x, y, z = vector
    return (x * cos_a - y * sin_a, x * sin_a + y * cos_a, z)


def read_lens_calibration(device: Device, socket, mode_size: tuple[int, int],
                          device_id: str = '') -> tuple[float, float, float] | None:
    """This unit's lens from its factory calibration: (focal px/rad, cx, cy) at `mode_size`,
    or None when the device carries none, which is not an error.

    The factory model is a pinhole with a rational distortion polynomial (`Perspective`, 14
    coefficients). Its `fx` is the scale at the optical axis, and on these lenses the fitted
    distortion keeps radius linear in angle to 0.5 % out to 75 degrees — so `fx` IS the
    equidistant focal, and `(cx, cy)` the optical centre, which is all `source_lens` needs.
    `getCameraIntrinsics` at the mode size already accounts for the 720-row centre crop (it
    shifts `cy` by 40).

    The warp is NOT built from this: the shared lens in the preset is. This is read so each
    unit's deviation from that shared lens can be reported (`lens_deviation`), and so the
    sensor variant is confirmed — the OAK-D Pro W ships with lenses 32 degrees apart, and
    `lens_field` of `fx` tells them apart (129 vs ~97).
    """
    try:
        calibration = device.readCalibration()
        matrix = calibration.getCameraIntrinsics(socket, mode_size[0], mode_size[1])
        model = calibration.getDistortionModel(socket)
        spec_fov: float = float(calibration.getFov(socket, useSpec=True))
        fx, fy = float(matrix[0][0]), float(matrix[1][1])
        cx, cy = float(matrix[0][2]), float(matrix[1][2])
        logger.info(f'{device_id} calibration {socket.name}: model {model}, spec fov {spec_fov:.1f} deg, '
                    f'fx {fx:.1f} fy {fy:.1f} px/rad, centre ({cx:.1f}, {cy:.1f}) '
                    f'at {mode_size[0]}x{mode_size[1]}')
        return fx, cx, cy
    except Exception as exc:                        # never let diagnostics break an open
        logger.debug(f'{device_id} could not read calibration: {exc}')
        return None


def imu_rotation_to_camera(device: Device, socket, device_id: str = '') -> list[list[float]] | None:
    """The IMU-to-camera matrix for this socket, or None if the board does not relate them.

    Logged rather than swallowed: whether this matrix exists decides whether the readout is
    running on the board's own figures or on `IMU_BOARD_ROLL`, and that is worth knowing from the
    log instead of inferring it from a suspicious roll reading — which is how the quarter turn was
    found in the first place.
    """
    try:
        matrix = device.readCalibration().getImuToCameraExtrinsics(socket)
        rotation = [[round(value, 4) for value in row[:3]] for row in matrix[:3]]
        logger.info(f'{device_id} IMU-to-camera rotation {rotation}')
        return matrix
    except Exception as exc:
        logger.info(f'{device_id} no IMU-to-camera extrinsics ({exc}) — '
                    f'falling back to the board constant, roll offset {IMU_BOARD_ROLL:.0f} deg')
        return None