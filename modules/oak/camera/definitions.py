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
MONO_RESOLUTION: MonoCameraProperties.SensorResolution = MonoCameraProperties.SensorResolution.THE_720_P
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