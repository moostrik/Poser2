from dataclasses import dataclass
from typing import Callable, TypeAlias

import torch

from modules.pose.frame import FrameDict


@dataclass(frozen=True)
class SegmentationImage:
    """GPU-resident segmentation result for a single tracked pose.

    Produced by MaskBatchExtractor after RVM inference.
    Format: CHW float16 RGB [0,1] for foreground; HW float16 [0,1] for mask.

    Attributes:
        track_id: Tracklet identifier
        mask: Segmentation mask on GPU (H, W) float16 [0,1]
        foreground: Masked foreground on GPU (3, H, W) float16 RGB CHW [0,1]
    """
    track_id: int
    mask: torch.Tensor
    foreground: torch.Tensor


SegmentationImageDict: TypeAlias = dict[int, SegmentationImage]
SegmentationImageCallback: TypeAlias = Callable[[FrameDict, SegmentationImageDict], None]
