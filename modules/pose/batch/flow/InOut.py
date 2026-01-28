# Standard library imports
from dataclasses import dataclass, field
from typing import Callable

# Third-party imports
import torch


@dataclass
class OpticalFlowInput:
    """Batch of consecutive GPU frame pairs for optical flow computation."""
    batch_id: int
    gpu_image_pairs: list[tuple[torch.Tensor, torch.Tensor]] = field(default_factory=list)  # List of (prev_crop, curr_crop) GPU tensors
    tracklet_ids: list[int] = field(default_factory=list)


@dataclass
class OpticalFlowOutput:
    """Results from optical flow computation. processed=False indicates batch was dropped."""
    batch_id: int
    flow_tensor: torch.Tensor | None = None  # GPU tensor (B, 2, H, W) FP32, flow field (x, y)
    tracklet_ids: list[int] = field(default_factory=list)
    processed: bool = True
    inference_time_ms: float = 0.0
    lock_time_ms: float = 0.0


OpticalFlowOutputCallback = Callable[[OpticalFlowOutput], None]