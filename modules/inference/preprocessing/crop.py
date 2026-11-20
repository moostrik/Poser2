from dataclasses import dataclass
from typing import Callable, TypeAlias

import torch

from modules.pose import FrameDict
from modules.utils import Rect


@dataclass(frozen=True)
class Image:
    """GPU-resident cropped region for a single tracked pose.

    Format: float16 RGB CHW [0, 1].

    Attributes:
        crop:      Resized crop on GPU (3, H, W) float16 RGB CHW [0, 1]
        ready:     Recorded on the crop stream after the crops are written; a model waits on it
                   (`stream.wait_event`) first, and it has passed for crops handed on finished. None for crops
                   already written.
        box:       The window the crop was cut at, uv on the camera image (rows down); None for a crop not cut
                   by the crop step.
    """
    crop: torch.Tensor
    ready: torch.cuda.Event | None = None
    box: Rect | None = None


CropImageDict: TypeAlias = dict[int, Image]
ImageCallback: TypeAlias = Callable[[FrameDict, CropImageDict], None]
AskedCallback: TypeAlias = Callable[[float, int, FrameDict], None]      # (timestamp, cam_id, the poses cut for that camera's image)
