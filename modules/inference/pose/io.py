# Standard library imports
from dataclasses import dataclass, field
from typing import Callable

# Third-party imports
import numpy as np
import torch


@dataclass
class DetectionInput:
    """Batch of GPU images for pose detection.

    GPU images will be resized to model dimensions on GPU.
    """
    batch_id: int
    gpu_images: list[torch.Tensor] = field(default_factory=list)  # GPU tensors (H, W, 3) RGB uint8

@dataclass
class DetectionOutput:
    """Results from pose detection. processed=False indicates batch was dropped."""
    batch_id: int
    point_batch: list[np.ndarray] = field(default_factory=list)   # List of (num_keypoints, 2) arrays, normalized [0, 1]
    score_batch: list[np.ndarray] = field(default_factory=list)   # List of (num_keypoints,) arrays, confidence scores [0, 1]
    processed: bool = True          # False if batch was dropped before processing
    inference_time_ms: float = 0.0  # For monitoring
    lock_time_ms: float = 0.0       # Time spent waiting for lock

PoseDetectionOutputCallback = Callable[[DetectionOutput], None]