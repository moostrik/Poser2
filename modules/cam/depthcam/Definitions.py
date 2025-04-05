import numpy as np
from enum import Enum
from typing import Callable
from depthai import Tracklet, TrackerType, ImgDetection, Rect, Point3f, Device


DETECTION_MODEL5S: str = "mobilenet-ssd_openvino_2021.4_5shave.blob"
DETECTION_MODEL6S: str = "mobilenet-ssd_openvino_2021.4_6shave.blob"
DETECTION_THRESHOLD: float = 0.5
TRACKER_TYPE: TrackerType = TrackerType.ZERO_TERM_IMAGELESS
# ZERO_TERM_COLOR_HISTOGRAM higher accuracy (but can drift when losing object)
# ZERO_TERM_IMAGELESS slightly faster
DEPTH_TRACKER_BOX_SCALE: float = 0.5
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


FrameCallback = Callable[[int, FrameType, np.ndarray, int], None]
DetectionCallback = Callable[[int, ImgDetection], None]
TrackerCallback = Callable[[int, Tracklet], None]
FPSCallback = Callable[[int, float], None]

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