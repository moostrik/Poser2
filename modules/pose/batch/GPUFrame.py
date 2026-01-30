# Standard library imports
from dataclasses import dataclass
from typing import Callable, TYPE_CHECKING

# Third-party imports
import torch

# Local application imports
from modules.pose.Frame import FrameDict


@dataclass(frozen=True)
class GPUFrame:
    """GPU-resident frame data for a single tracklet.

    Contains the full source image and a cropped region on GPU memory.
    Used to pass GPU data directly to TRT inference classes, avoiding
    redundant CPU→GPU transfers.

    Attributes:
        track_id: Tracklet identifier
        full_image: Full source frame on GPU (H, W, 3) float32 BGR [0,1]
        crop: Cropped and resized region on GPU (crop_height, crop_width, 3) float32 RGB [0,1]
        prev_crop: Previous frame cropped at CURRENT bbox location for optical flow.
                   float32 RGB [0,1], None if no previous frame available.
        mask: Optional segmentation mask on GPU (mask_height, mask_width) float16 or float32 [0,1]
    """
    track_id: int
    full_image: torch.Tensor
    crop: torch.Tensor
    prev_crop: torch.Tensor | None = None
    mask: torch.Tensor | None = None
    foreground: torch.Tensor | None = None


# Type aliases
GPUFrameDict = dict[int, GPUFrame]
GPUFrameCallback = Callable[[FrameDict, GPUFrameDict], None]
