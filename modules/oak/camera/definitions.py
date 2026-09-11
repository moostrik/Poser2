# https://blobconverter.luxonis.com/

import numpy as np
from enum import Enum, IntEnum, auto
from typing import Callable, TypeAlias
from depthai import (Tracklet, TrackerType, ImgDetection, Rect, Point3f, Device,
                     SpatialLocationCalculatorAlgorithm, MonoCameraProperties, ColorCameraProperties)

import logging
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Frame geometry — the one source for every camera dimension
# ---------------------------------------------------------------------------
# The sensor mode is the only knob; every width, height, aspect ratio and buffer
# size in the pipeline and in the render derives from it. Nothing downstream may
# restate a frame dimension as a literal — that is how the two silently drift.

# `dai.node.Warp` needs both output dimensions divisible by this.
WARP_ALIGNMENT: int = 16

MONO_SIZES: dict[MonoCameraProperties.SensorResolution, tuple[int, int]] = {
    MonoCameraProperties.SensorResolution.THE_400_P: ( 640, 400),
    MonoCameraProperties.SensorResolution.THE_480_P: ( 640, 480),
    MonoCameraProperties.SensorResolution.THE_720_P: (1280, 720),
    MonoCameraProperties.SensorResolution.THE_800_P: (1280, 800),
}

# The mono sensor mode the installation runs. OV9282 W is natively 1280x800;
# THE_720_P is a pure vertical crop of it, so the horizontal field — and with it the
# whole column-to-azimuth mapping — is identical either way. Change this one line to
# change the resolution everywhere.
#
# NOT derived from it, and to be edited by hand alongside it, because they are set by eye
# against the frame rather than computed:
#   camera.tracker.min_height             a fraction of frame height, so it scales
#   camera.tracker.seam.max_height_diff   idem
#   pose.distance_extractor.near_y/far_y  positions in the frame, so they scale and shift
#   render.py's 'track' row src_aspect_ratio
# (The vertical field is NOT in that list — see `frame_fov` below, which derives it.)
MONO_RESOLUTION: MonoCameraProperties.SensorResolution = MonoCameraProperties.SensorResolution.THE_800_P
MONO_SIZE: tuple[int, int] = MONO_SIZES[MONO_RESOLUTION]

# Colour preview sizes, already trimmed to the warp's alignment: 1080 is not divisible
# by 16, so the 1080p preview is cropped to 1072 rows.
COLOR_SIZES: dict[ColorCameraProperties.SensorResolution, tuple[int, int]] = {
    ColorCameraProperties.SensorResolution.THE_720_P:  (1280,  720),
    ColorCameraProperties.SensorResolution.THE_1080_P: (1920, 1072),
}


def mono_frame_size(square: bool = False) -> tuple[int, int]:
    """The mono frame as it leaves the pipeline. `square` crops it to height x height."""
    width, height = MONO_SIZE
    return (height, height) if square else (width, height)


def color_resolution(do_720p: bool = False) -> ColorCameraProperties.SensorResolution:
    """The colour sensor mode. Unlike mono this is chosen at runtime, by the `hd_ready`
    setting — a shared camera-module setting, so it stays a setting. The choice itself lives
    here, once, so the pipeline and the size helpers cannot disagree about it."""
    return (ColorCameraProperties.SensorResolution.THE_720_P if do_720p
            else ColorCameraProperties.SensorResolution.THE_1080_P)


def color_frame_size(do_720p: bool = False, square: bool = False) -> tuple[int, int]:
    """The colour frame as it leaves the pipeline. `square` crops it to height x height."""
    width, height = COLOR_SIZES[color_resolution(do_720p)]
    return (height, height) if square else (width, height)


def frame_size(color: bool, do_720p: bool = False, square: bool = False) -> tuple[int, int]:
    """The frame size a camera configuration produces, whichever path it takes. Anything that
    needs the camera's aspect ratio — the render's layout, a texture allocation — asks here
    instead of restating it."""
    return color_frame_size(do_720p, square) if color else mono_frame_size(square)


def mode_size(color: bool, do_720p: bool = False) -> tuple[int, int]:
    """The frame size BEFORE any square crop — the size the sensor mode actually produces.
    This is the width the `fov` setting is quoted against, so it is what `degrees_per_pixel`
    must be given."""
    return frame_size(color, do_720p, square=False)


# ---------------------------------------------------------------------------
#  Lens geometry — one stored number per camera, everything else derived
# ---------------------------------------------------------------------------
# Sensor reference, for choosing the `fov` setting. EVERY figure below is for the sensor's
# FULL readout, so it is true only in the mode that reads every line. Pairing a field with
# the wrong mode is how `vfov` went stale at 720 rows.
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
# `get_device_list(verbose=True)` logs the sensor behind each socket, so the hardware
# answers the "which variant is this?" question itself rather than it being inferred.
#
# Both lenses are near-equidistant — angle maps linearly to radius — so degrees-per-pixel
# is the same on both axes and the vertical field never needs storing. The two published
# triples confirm that independently of each other:
#
#   OV9282   127 * 800/1280  = 79.4   published 79.5
#   IMX378    95 * 3040/4056 = 71.2   published 72
#
# So only the horizontal field is a setting. Everything else comes off the frame.


def degrees_per_pixel(fov_h: float, mode_width: int) -> float:
    """Angular size of one pixel (degrees).

    ``mode_width`` must be the width of the UN-cropped frame the sensor mode produces,
    because ``fov_h`` is quoted for that full width. A later crop — the square crop, or the
    1080 -> 1072 trim — removes pixels without changing this scale, which is exactly why the
    scale rather than a field angle is the thing worth deriving from.
    """
    if mode_width <= 0:
        return 0.0
    return fov_h / float(mode_width)


# Mesh resolution for the tilt warp. See `tilt_mesh_points` for why 2 is not enough and
# why 32 is where this stops mattering.
WARP_MESH: int = 32


def frame_fov(fov_h: float, mode_width: int, out_size: tuple[int, int]) -> tuple[float, float]:
    """The horizontal and vertical field (degrees) a delivered frame actually covers.

    ``mode_width`` is the un-cropped width ``fov_h`` was quoted against; ``out_size`` is the
    frame as it leaves the pipeline. Crops shrink the field in proportion, so this holds
    across the square crop and every resolution change without a second stored number.
    """
    dpp: float = degrees_per_pixel(fov_h, mode_width)
    width, height = out_size
    return dpp * width, dpp * height


def tilt_mesh_points(
    src_size: tuple[int, int],
    out_size: tuple[int, int],
    mode_width: int,
    fov_h: float,
    tilt: float,
    flip_h: bool = False,
    flip_v: bool = False,
    mesh_w: int = WARP_MESH,
    mesh_h: int = WARP_MESH,
) -> list[tuple[float, float]]:
    """Warp mesh that undoes a camera's up-tilt: the source pixel each output grid point reads.

    WHY THIS IS NOT A HOMOGRAPHY. These lenses are near-equidistant — distance from the frame
    centre is proportional to the ANGLE off the optical axis, not to its tangent. A pinhole
    lens uses the tangent, and that stretch is exactly what makes a tilted pinhole view a
    trapezoid with straight edges, which `cv2.getPerspectiveTransform` reproduces. With no
    stretch nothing cancels: tilting bends straight lines into curves. At 15 deg on a 127 deg
    frame the centre of a row moves 151 px while its ends move 74 px, a 77 px bow. So the mesh
    is built by unprojecting each output pixel to a ray, rotating the ray, and reprojecting.

    WHY THE MESH IS 32 x 32. The Warp node interpolates linearly between mesh points, so
    2 columns can only express a straight source line per row — which is the very family the
    homography already spanned, and the bow above is what the correction IS. This is a
    threshold, not a spectrum: 2 is unusable, ~16 is already sub-pixel, and 32 x 32 differs
    from 64 x 64 by under 0.3 px even at 30 deg. Error grows linearly with frame width and
    falls with the square of the mesh count, so 32 still holds to 0.23 px on a 4056 px frame.

    `src_size` is the frame the sensor delivers and `out_size` the warp's output, which is
    smaller only for the square crop (taken from the centre). `mode_width` is the un-cropped
    width `fov_h` is quoted against. Positive `tilt` means the camera is aimed UP, so the
    corrected view reads from lower in the source frame.

    At `tilt == 0` the result is exactly the identity grid (or an exact mirror under the
    flips), with no floating-point round-trip. Nothing changes until an angle is set.
    """
    src_w, src_h = src_size
    out_w, out_h = out_size
    x_off: float = (src_w - out_w) / 2.0          # the square crop is cut from the centre
    y_off: float = (src_h - out_h) / 2.0
    # Pixel-index convention: a frame of width W spans 0..W-1, so its optical axis sits at
    # (W-1)/2, not W/2. The half pixel matters — it is what keeps the centre exactly on axis.
    cx, cy = (src_w - 1) / 2.0, (src_h - 1) / 2.0

    dpp: float = np.radians(degrees_per_pixel(fov_h, mode_width))
    identity: bool = tilt == 0.0 or dpp <= 0.0
    t: float = np.radians(-tilt)                   # undo the tilt, so negate it
    cos_t, sin_t = float(np.cos(t)), float(np.sin(t))

    points: list[tuple[float, float]] = []
    for gy in np.linspace(0.0, out_h - 1.0, mesh_h):
        for gx in np.linspace(0.0, out_w - 1.0, mesh_w):
            ox: float = (out_w - 1.0 - gx) if flip_h else float(gx)
            oy: float = (out_h - 1.0 - gy) if flip_v else float(gy)
            x, y = x_off + ox, y_off + oy
            if identity:
                points.append((x, y))
                continue
            dx, dy = x - cx, y - cy
            phi: float = float(np.hypot(dx, dy)) * dpp
            psi: float = float(np.arctan2(dy, dx))
            sin_p: float = float(np.sin(phi))
            d0, d1, d2 = sin_p * np.cos(psi), sin_p * np.sin(psi), float(np.cos(phi))
            d1, d2 = cos_t * d1 - sin_t * d2, sin_t * d1 + cos_t * d2
            r2: float = float(np.arccos(np.clip(d2, -1.0, 1.0))) / dpp
            psi2: float = float(np.arctan2(d1, d0))
            points.append((cx + r2 * float(np.cos(psi2)), cy + r2 * float(np.sin(psi2))))
    return points


YOLOV8_WIDE_5S: str = "yolov8n_coco_640x352_5S.blob"
YOLOV8_WIDE_6S: str = "yolov8n_coco_640x352_6S.blob"
YOLOV8_WIDE_7S: str = "yolov8n_coco_640x352_7S.blob"
YOLOV8_SQUARE_5S: str = "yolov8n_coco_416x416_5S.blob"
YOLOV8_SQUARE_6S: str = "yolov8n_coco_416x416_6S.blob"
YOLOV8_SQUARE_7S: str = "yolov8n_coco_416x416_7S.blob"
YOLO_CONFIDENCE_THRESHOLD: float = 0.66
YOLO_OVERLAP_THRESHOLD: float = 0.5

# The detector's input, fixed by the blob it was compiled for — the names above carry it.
# Not a camera dimension: the frame is resized into this whatever the sensor mode is.
DETECTOR_INPUT_WIDE:   tuple[int, int] = (640, 352)
DETECTOR_INPUT_SQUARE: tuple[int, int] = (416, 416)

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