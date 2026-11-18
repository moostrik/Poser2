from dataclasses import dataclass
from typing import Callable, TypeAlias

import numpy as np

from ..preprocessing import GpuImage
from .measures import HeadPose


@dataclass(frozen=True)
class Face:
    """One face of an image, in the image's pixels: its box, the detector's score and five points (eyes, nose,
    mouth corners), the 106 landmarks, and what they measure."""
    bbox: np.ndarray        # x0, y0, x1, y1
    score: float
    kps: np.ndarray         # 5 x 2
    landmarks: np.ndarray   # 106 x 2
    pose: HeadPose
    eyes: float             # openness of the less open eye
    eye_dist: float         # distance between the eyes, relative to the image width

    @property
    def width(self) -> float:
        return float(self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return float(self.bbox[3] - self.bbox[1])


@dataclass(frozen=True)
class FaceFrame:
    """The faces of one uploaded image, largest score first, with the image they were found in (its CPU frame
    with it, `image.frame`), carried unread, and its time."""
    cam_id: int
    timestamp: float
    width: int
    height: int
    faces: tuple[Face, ...]
    image: GpuImage


FaceFrameCallback: TypeAlias = Callable[[FaceFrame], None]
