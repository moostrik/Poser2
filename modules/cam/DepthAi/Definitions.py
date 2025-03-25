import numpy as np
from enum import Enum
from typing import Callable
from depthai import Tracklet, ImgDetection, Rect, Point3f
# Tracklet = T
# Detection = D
# Detections = list[Detection]


class FrameType(Enum):
    NONE =  0
    VIDEO = 1
    LEFT =  2
    RIGHT = 3
    STEREO= 4

FrameTypeNames: list[str] = [e.name for e in FrameType]

FrameCallback = Callable[[int, FrameType, np.ndarray], None]
PreviewCallback = Callable[[int, np.ndarray], None]
DetectionCallback = Callable[[int, ImgDetection], None]
TrackerCallback = Callable[[int, Tracklet], None]
FPSCallback = Callable[[int, float], None]


exposureRange:          tuple[int, int] = (1000, 33000)
isoRange:               tuple[int, int] = ( 100, 1600 )
balanceRange:           tuple[int, int] = (1000, 12000)
contrastRange:          tuple[int, int] = ( -10, 10   )
brightnessRange:        tuple[int, int] = ( -10, 10   )
lumaDenoiseRange:       tuple[int, int] = (   0, 4    )
saturationRange:        tuple[int, int] = ( -10, 10   )
sharpnessRange:         tuple[int, int] = (   0, 4    )

stereoDepthRange:       tuple[int, int] = ( 500, 15000)
stereoBrightnessRange:  tuple[int, int] = (   0, 255  )

class StereoMedianFilterType(Enum):
    OFF = 0
    KERNEL_3x3 = 1
    KERNEL_5x5 = 2
    KERNEL_7x7 = 3

StereoMedianFilterTypeNames: list[str] = [e.name for e in StereoMedianFilterType]

# class Tracklet():
#     class TrackingStatus(Enum):
#         NEW     = 0
#         TRACKED = 1
#         LOST    = 2
#         REMOVED = 3

    # def __init__(self, cam_id: int,
    #              x: float, y: float, width: float, height: float,
    #              track_id: int, status: TrackingStatus, age: int, confidence: float,
    #              sp_x: float | None, sp_y: float | None, sp_z: float | None) -> None:
    #     self.cam_id: int = cam_id
    #     self.centre: float = x + width / 2
    #     self.x: float = x
    #     self.y: float = y
    #     self.width: float = width
    #     self.height: float = height
    #     self.track_id: int = track_id
    #     self.status: Tracklet.TrackingStatus = status
    #     self.age: int = age
    #     self.confidence: float = confidence
    #     self.sp_x: float | None = sp_x
    #     self.sp_y: float | None = sp_y
    #     self.sp_z: float | None = sp_z

    #     image: np.ndarray | None = None


    # @staticmethod
    # def from_dai(tracklet: T, cam_id: int) -> 'Tracklet':
    #     roi: Rect = tracklet.roi
    #     sp: Point3f = tracklet.spatialCoordinates
    #     has_sp: bool = sp.x + sp.y + sp.z != 0
    #     sp_x: float | None = None
    #     sp_y: float | None = None
    #     sp_z: float | None = None
    #     if has_sp:
    #         sp_x = sp.x
    #         sp_y = sp.y
    #         sp_z = sp.z
    #     return Tracklet(
    #         cam_id = cam_id,
    #         x = roi.x,
    #         y = roi.y,
    #         width = roi.width,
    #         height = roi.height,
    #         track_id = tracklet.id,
    #         status = Tracklet.TrackingStatus(tracklet.status.value),
    #         age = tracklet.age,
    #         confidence = tracklet.srcImgDetection.confidence,
    #         sp_x = sp_x,
    #         sp_y = sp_y,
    #         sp_z = sp_z
    #     )

# Tracklets = list[T]