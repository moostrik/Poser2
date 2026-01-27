# Standard library imports
from dataclasses import dataclass, field
from typing import Callable

# Third-party imports
import numpy as np
import torch


@dataclass
class SegmentationInput:
    """Batch of images for segmentation. Images should be (H, W, 3) BGR uint8."""
    batch_id: int
    images: list[np.ndarray]
    tracklet_ids: list[int] = field(default_factory=list)  # Corresponding tracklet IDs for each image


@dataclass
class SegmentationOutput:
    """Results from segmentation. processed=False indicates batch was dropped."""
    batch_id: int
    mask_tensor: torch.Tensor | None = None    # GPU tensor (B, H, W) FP16, alpha matte [0, 1]
    fgr_tensor: torch.Tensor | None = None     # GPU tensor (B, 3, H, W) FP32, foreground RGB [0, 1]
    tracklet_ids: list[int] = field(default_factory=list)  # Corresponding tracklet IDs
    processed: bool = True          # False if batch was dropped before processing
    inference_time_ms: float = 0.0  # For monitoring
    lock_time_ms: float = 0.0       # Time spent waiting for lock


SegmentationOutputCallback = Callable[[SegmentationOutput], None]