from dataclasses import dataclass
import math
from typing import Callable, TypeAlias

import torch

from modules.pose import FrameDict


@dataclass(frozen=True)
class Image:
    """GPU-resident segmentation result for a single tracked pose.

    Produced by MaskBatchExtractor after RVM inference.
    Format: CHW float16 RGB [0,1] for foreground; HW float16 [0,1] for mask.

    Attributes:
        mask: Segmentation mask on GPU (H, W) float16 [0,1]
        foreground: Masked foreground on GPU (3, H, W) float16 RGB CHW [0,1]
        timestamp: The capture time of the camera frame it was made from (the pose's), the frame's identity; nan when unknown
    """
    mask: torch.Tensor
    foreground: torch.Tensor
    timestamp: float = math.nan


ImageDict: TypeAlias = dict[int, Image]
ImageCallback: TypeAlias = Callable[[FrameDict, ImageDict], None]
