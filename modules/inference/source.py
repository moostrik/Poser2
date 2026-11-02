from dataclasses import dataclass
from typing import Callable, TypeAlias

import torch


ImageDict: TypeAlias = dict[int, torch.Tensor]
TimestampDict: TypeAlias = dict[int, float]     # each camera's image's capture time (`CameraFrame.timestamp`)


@dataclass(frozen=True)
class GpuImage:
    """One camera frame on the GPU as `Uploader` made it: float16 RGB CHW [0, 1], straightened and
    mirrored, with the frame's capture time. `ready` is recorded on the upload's stream after the
    image is written: a reader on another stream waits on it (`stream.wait_event(ready)`) first."""
    cam_id: int
    timestamp: float
    image: torch.Tensor
    ready: torch.cuda.Event


GpuImageCallback: TypeAlias = Callable[[GpuImage], None]
