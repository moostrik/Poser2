# Standard library imports
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import cupy as cp


@dataclass
class GPUFrame:
    """GPU-resident frame data for a single tracklet.

    Contains the full source image and a cropped region on GPU memory.
    Used to pass GPU data directly to TRT inference classes, avoiding
    redundant CPU→GPU transfers.

    Attributes:
        track_id: Tracklet identifier
        full_image: Full source frame on GPU (H, W, 3) BGR uint8
        crop: Cropped and resized region on GPU (crop_height, crop_width, 3) BGR uint8
        prev_crop: Previous frame cropped at CURRENT bbox location for optical flow.
                   None if no previous frame available.
    """
    track_id: int
    full_image: 'cp.ndarray'
    crop: 'cp.ndarray'
    prev_crop: 'cp.ndarray | None' = None


# Type aliases
GPUFrameDict = dict[int, GPUFrame]
