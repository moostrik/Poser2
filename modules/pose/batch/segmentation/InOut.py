# Standard library imports
from dataclasses import dataclass, field
from typing import Callable, TYPE_CHECKING

# Third-party imports
import torch

if TYPE_CHECKING:
    import cupy as cp


@dataclass
class SegmentationInput:
    """Batch of GPU images for segmentation.

    GPU images will be processed directly on GPU.
    """
    batch_id: int
    gpu_images: 'list[cp.ndarray]' = field(default_factory=list)  # GPU images (H, W, 3) RGB uint8
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