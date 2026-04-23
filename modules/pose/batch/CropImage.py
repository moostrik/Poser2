from dataclasses import dataclass
from typing import Callable, TypeAlias

import torch

from modules.pose.frame import FrameDict


@dataclass(frozen=True)
class CropImage:
    """GPU-resident cropped region for a single tracked pose.

    Produced by ImageCropProcessor once per tracked pose per frame.
    Format: CHW float16 RGB [0,1].

    Attributes:
        track_id: Tracklet identifier
        crop: Resized crop on GPU (3, H, W) float16 RGB CHW [0,1]
        prev_crop: Previous frame cropped at current bbox location for optical flow.
                   (3, H, W) float16 RGB CHW [0,1], None if no previous frame available.
    """
    track_id: int
    crop: torch.Tensor
    prev_crop: torch.Tensor | None = None


CropImageDict: TypeAlias = dict[int, CropImage]
CropImageCallback: TypeAlias = Callable[[FrameDict, CropImageDict], None]
