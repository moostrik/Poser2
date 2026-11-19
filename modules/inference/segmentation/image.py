from dataclasses import dataclass
import math
from typing import Callable, TypeAlias

import torch

from modules.pose import FrameDict
from modules.utils import Rect


@dataclass(frozen=True)
class Image:
    """GPU-resident segmentation result for a single tracked pose.

    Made by the `Segmenter`, Robust Video Matting's alpha matte.
    Format: CHW float16 RGB [0,1] for foreground; HW float16 [0,1] for mask.

    Attributes:
        mask: Segmentation mask on GPU (H, W) float16 [0,1]
        foreground: Masked foreground on GPU (3, H, W) float16 RGB CHW [0,1]
        timestamp: The capture time of the camera frame it was made from (the pose's), the frame's identity; nan when unknown
        ready: Recorded on the segmentation's stream after the mask and the foreground are written: a reader on
               another stream waits on it (`stream.wait_event(ready)`) first; None where nothing is queued
        box: The window its crop was cut at, uv on the camera image (rows down), where the mask lies in the camera
             image; None when its crop carried none
    """
    mask: torch.Tensor
    foreground: torch.Tensor
    timestamp: float = math.nan
    ready: torch.cuda.Event | None = None
    box: Rect | None = None


ImageDict: TypeAlias = dict[int, Image]
ImageCallback: TypeAlias = Callable[[FrameDict, ImageDict], None]
