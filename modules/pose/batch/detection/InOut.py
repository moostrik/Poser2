# Standard library imports
from dataclasses import dataclass, field
from typing import Callable

# Third-party imports
import numpy as np


@dataclass
class DetectionInput:
    """Batch of images for pose detection. Images must be 192x256 (HxW)."""
    batch_id: int
    images: list[np.ndarray]

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