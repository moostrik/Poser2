# https://blobconverter.luxonis.com/

import numpy as np
from enum import Enum, IntEnum, auto
from typing import Callable, TypeAlias
from depthai import Tracklet, TrackerType, ImgDetection, Rect, Point3f, Device

import logging
logger = logging.getLogger(__name__)


YOLOV8_WIDE_5S: str = "yolov8n_coco_640x352_5S.blob"
YOLOV8_WIDE_6S: str = "yolov8n_coco_640x352_6S.blob"
YOLOV8_WIDE_7S: str = "yolov8n_coco_640x352_7S.blob"
YOLOV8_SQUARE_5S: str = "yolov8n_coco_416x416_5S.blob"
YOLOV8_SQUARE_6S: str = "yolov8n_coco_416x416_6S.blob"
YOLOV8_SQUARE_7S: str = "yolov8n_coco_416x416_7S.blob"
YOLO_CONFIDENCE_THRESHOLD: float = 0.66
YOLO_OVERLAP_THRESHOLD: float = 0.5

TRACKER_PERSON_LABEL: int = 0
TRACKER_TYPE: TrackerType = TrackerType.ZERO_TERM_IMAGELESS
# ZERO_TERM_COLOR_HISTOGRAM higher accuracy (but can drift when losing object)
# ZERO_TERM_IMAGELESS slightly faster

class FrameType(Enum):
    NONE_ = 0
    VIDEO = 1

FRAME_TYPE_NAMES: list[str] = [e.name for e in FrameType]

FRAME_TYPE_LABEL_DICT: dict[FrameType, str] = {
    FrameType.NONE_: 'N',
    FrameType.VIDEO: 'C',
}

EXPOSURE_RANGE:     tuple[int, int] = (1000, 33000)
ISO_RANGE:          tuple[int, int] = ( 100, 1600 )
BALANCE_RANGE:      tuple[int, int] = (1000, 12000)
CONTRAST_RANGE:     tuple[int, int] = ( -10, 10   )
BRIGHTNESS_RANGE:   tuple[int, int] = ( -10, 10   )
LUMA_DENOISE_RANGE: tuple[int, int] = (   0, 4    )
SATURATION_RANGE:   tuple[int, int] = ( -10, 10   )
SHARPNESS_RANGE:    tuple[int, int] = (   0, 4    )

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
    VIDEO_FRAME_IN = auto()

class Output(IntEnum):
    VIDEO_FRAME_OUT = auto()
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