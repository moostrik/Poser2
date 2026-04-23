from dataclasses import dataclass
from typing import Callable, TypeAlias

import torch


@dataclass(frozen=True)
class CameraImage:
    """GPU-resident full camera image for a single camera.

    Produced by CropProcessor once per camera per frame.
    Format: CHW float16 RGB [0,1].

    Attributes:
        cam_id: Camera identifier
        image: Full source frame on GPU (3, H, W) float16 RGB CHW [0,1]
    """
    cam_id: int
    image: torch.Tensor


CameraImageDict: TypeAlias = dict[int, CameraImage]
CameraImageCallback: TypeAlias = Callable[[CameraImageDict], None]
