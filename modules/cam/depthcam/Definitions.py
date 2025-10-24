# https://blobconverter.luxonis.com/

import numpy as np
from enum import Enum, IntEnum, auto
from typing import Callable
from depthai import Tracklet, TrackerType, ImgDetection, Rect, Point3f, Device, SpatialLocationCalculatorAlgorithm


YOLOV8_WIDE_5S: str = "yolov8n_coco_640x352_5S.blob"
YOLOV8_WIDE_6S: str = "yolov8n_coco_640x352_6S.blob"
YOLOV8_WIDE_7S: str = "yolov8n_coco_640x352_7S.blob"
YOLOV8_SQUARE_5S: str = "yolov8n_coco_416x416_5S.blob"
YOLOV8_SQUARE_6S: str = "yolov8n_coco_416x416_6S.blob"
YOLOV8_SQUARE_7S: str = "yolov8n_coco_416x416_7S.blob"
YOLO_CONFIDENCE_THRESHOLD: float = 0.5
YOLO_OVERLAP_THRESHOLD: float = 0.5

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

FrameCallback = Callable[[int, FrameType, np.ndarray], None]
SyncCallback = Callable[[int, dict[FrameType, np.ndarray], float], None]
DetectionCallback = Callable[[int, ImgDetection], None]
TrackerCallback = Callable[[int, list[Tracklet]], None]
FPSCallback = Callable[[int, float], None]

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
        print('-- CAMERAS --------------------------------------------------')
    for device in Device.getAllAvailableDevices():
        device_list.append(device.getMxId())
        if verbose:
            print(f"Camera: {device.getMxId()} {device.state}")
    if verbose:
        print('-------------------------------------------------------------')
    return device_list