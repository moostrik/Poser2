from dataclasses import dataclass
from typing import Callable, TypeAlias

import torch

from modules.camera import CameraFrame


ImageDict: TypeAlias = dict[int, torch.Tensor]
TimestampDict: TypeAlias = dict[int, float]     # each camera's image's capture time (`CameraFrame.timestamp`)

# What an uploader handed on at once: each camera's image by cam_id, the images' capture times, and the event
# recorded after them on the upload's stream, which a reader on another stream waits on first
ImageSnapshot: TypeAlias = tuple[ImageDict, TimestampDict, torch.cuda.Event]
ImageSnapshotCallback: TypeAlias = Callable[[ImageSnapshot], None]
CaptureCallback: TypeAlias = Callable[[TimestampDict], None]     # an uploaded set's capture times, no images


@dataclass(frozen=True)
class GpuImage:
    """One camera frame on the GPU as an uploader made it: float16 RGB CHW [0, 1], straightened and
    mirrored, with the frame's capture time. `ready` is recorded on the upload's stream after the
    image is written: a reader on another stream waits on it (`stream.wait_event(ready)`) first; it has passed for
    an image handed on finished.
    `frame` is the camera frame it was uploaded from, so the image and the CPU frame of one moment travel as
    one; None where nobody uploaded one."""
    cam_id: int
    timestamp: float
    image: torch.Tensor
    ready: torch.cuda.Event
    frame: CameraFrame | None = None


GpuImageCallback: TypeAlias = Callable[[GpuImage], None]


@dataclass(frozen=True)
class GpuDepth:
    """One camera's depth frame on the GPU as the `SerialUploader` made it: float32 (H, W, 2), meters and the confidence
    0 to 1, both 0 where there is no depth, the confidence 1 where the frame has none; on the color image's view at a
    grid as fine as the depth itself (`aligned_depth_size`), rows down, straightened and mirrored as the image is, with
    the frame's capture time. `ready` as a `GpuImage`'s."""
    cam_id: int
    timestamp: float
    depth: torch.Tensor
    ready: torch.cuda.Event


GpuDepthCallback: TypeAlias = Callable[[GpuDepth], None]
