# Standard library imports
from dataclasses import dataclass, field
from typing import Callable

# Third-party imports
import numpy as np
import torch


@dataclass
class OpticalFlowInput:
    """Batch of consecutive frame pairs for optical flow computation."""
    batch_id: int
    frame_pairs: list[tuple[np.ndarray, np.ndarray]]  # List of (prev_frame, curr_frame) pairs
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