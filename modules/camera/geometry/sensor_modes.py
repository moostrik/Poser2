"""The sensor modes and the size of the frame each delivers."""

import math
from enum import IntEnum, auto

from .lens import output_focal, source_lens

import logging
logger = logging.getLogger(__name__)

WARP_ALIGNMENT: int = 16        # `dai.node.Warp` needs both output dimensions divisible by this


class CameraResolution(IntEnum):
    """The sensor mode, one label for both sensors: the `resolution` setting."""
    P720  = 0       # 1280 x 720   both sensors
    P800  = auto()  # 1280 x 800   both sensors; the OV9282 W's full readout
    P1080 = auto()  # 1920 x 1072  color only


# The delivered size of each mode, trimmed to the warp's alignment: 1080 rows are cropped to 1072.
RESOLUTION_SIZES: dict[CameraResolution, tuple[int, int]] = {
    CameraResolution.P720:  (1280,  720),
    CameraResolution.P800:  (1280,  800),
    CameraResolution.P1080: (1920, 1072),
}

MONO_RESOLUTIONS: frozenset[CameraResolution] = frozenset({CameraResolution.P720, CameraResolution.P800})
COLOR_RESOLUTIONS: frozenset[CameraResolution] = frozenset({CameraResolution.P720, CameraResolution.P800,
                                                             CameraResolution.P1080})

_FALLBACK_RESOLUTION: CameraResolution = CameraResolution.P800     # a mono camera asked for a mode it has not
_warned_resolutions: set[tuple[bool, CameraResolution]] = set()     # warned once each
_warned_heights: set[int] = set()                                   # warned once each


def resolve_resolution(color: bool, resolution: CameraResolution) -> CameraResolution:
    """The mode this sensor runs: `resolution`, or `P800` where the sensor has not that mode, warned once."""
    available = COLOR_RESOLUTIONS if color else MONO_RESOLUTIONS
    if resolution in available:
        return resolution
    if (color, resolution) not in _warned_resolutions:
        _warned_resolutions.add((color, resolution))
        logger.warning(
            "%s is not available on the %s sensor — falling back to %s",
            resolution.name, 'color' if color else 'mono', _FALLBACK_RESOLUTION.name,
        )
    return _FALLBACK_RESOLUTION


def aligned_height(height: int) -> int:
    """`height` rounded to the warp's alignment, warning once when that changes it."""
    aligned: int = int(round(height / WARP_ALIGNMENT)) * WARP_ALIGNMENT
    if aligned != height and height not in _warned_heights:
        _warned_heights.add(height)
        logger.warning("frame height %d is not a multiple of %d — using %d", height, WARP_ALIGNMENT, aligned)
    return max(WARP_ALIGNMENT, aligned)


def _delivered(mode: tuple[int, int], square: bool, height: int) -> tuple[int, int]:
    """The delivered frame for a sensor mode: height x height when `square`, else the mode's width by `height`
    rows (0: the mode's own)."""
    width, mode_height = mode
    if square:
        return (mode_height, mode_height)
    return (width, aligned_height(height) if height > 0 else mode_height)


def mono_frame_size(resolution: CameraResolution, square: bool = False, height: int = 0) -> tuple[int, int]:
    """The mono frame as it leaves the pipeline; `height` in px, 0 the mode's rows."""
    return _delivered(RESOLUTION_SIZES[resolve_resolution(False, resolution)], square, height)


def color_frame_size(resolution: CameraResolution, square: bool = False, height: int = 0) -> tuple[int, int]:
    """The color frame as it leaves the pipeline; `height` in px, 0 the mode's rows."""
    return _delivered(RESOLUTION_SIZES[resolve_resolution(True, resolution)], square, height)


def frame_size(color: bool, resolution: CameraResolution, square: bool = False,
               height: int = 0) -> tuple[int, int]:
    """The frame a camera configuration delivers; `height` is `delivered_height`'s."""
    return (color_frame_size(resolution, square, height) if color
            else mono_frame_size(resolution, square, height))


def delivered_height(color: bool, resolution: CameraResolution, fov_h: float, tilt: float,
                     lens_fov: float = 0.0, lens_center: tuple[float, float] = (0.0, 0.0)) -> int:
    """The delivered frame's rows: the sensor's full reach at this tilt (`full_frame_height`)."""
    src: tuple[int, int] = mode_size(color, resolution)
    return full_frame_height(src, src[0], fov_h, tilt, lens_fov, lens_center)


def full_frame_height(src_size: tuple[int, int], mode_width: int, fov_h: float, tilt: float,
                      lens_fov: float = 0.0,
                      lens_center: tuple[float, float] = (0.0, 0.0)) -> int:
    """The aligned frame height at which the center column carries every sensor row."""
    focal: float = output_focal(fov_h, mode_width)
    lens_focal, _, cy = source_lens(src_size, mode_width, fov_h, lens_fov, lens_center)
    if focal <= 0.0 or lens_focal <= 0.0:
        return src_size[1]
    bottom: float = math.radians(tilt) - (src_size[1] - 1 - cy) / lens_focal
    top: float = math.radians(tilt) + cy / lens_focal
    rows: float = focal * (math.tan(top) - math.tan(bottom)) + 1.0
    return int(math.ceil(rows / WARP_ALIGNMENT)) * WARP_ALIGNMENT


def mode_size(color: bool, resolution: CameraResolution) -> tuple[int, int]:
    """The frame the sensor mode produces, before a square crop: the width `fov` is quoted against."""
    return frame_size(color, resolution, square=False)
