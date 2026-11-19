from dataclasses import dataclass
from typing import Callable, TypeAlias

import torch

from modules.pose import FrameDict
from modules.settings import BaseSettings, Field
from modules.utils import Rect


@dataclass(frozen=True)
class Image:
    """GPU-resident cropped region for a single tracked pose.

    Format: float16 RGB CHW [0, 1].

    Attributes:
        crop:      Resized crop on GPU (3, H, W) float16 RGB CHW [0, 1]
        prev_crop: Previous frame cropped at current bbox for optical flow,
                   or None if no previous frame is available.
        ready:     Recorded on the crop stream after the crops are written; a reader on another stream
                   waits on it (`stream.wait_event`) first. None for crops already written.
        box:       The window the crop was cut at, uv on the camera image (rows down); None for a crop not cut
                   by the crop step.
    """
    crop: torch.Tensor
    prev_crop: torch.Tensor | None = None
    ready: torch.cuda.Event | None = None
    box: Rect | None = None


CropImageDict: TypeAlias = dict[int, Image]
ImageCallback: TypeAlias = Callable[[FrameDict, CropImageDict], None]
AskedCallback: TypeAlias = Callable[[float, int, FrameDict], None]      # (timestamp, cam_id, the poses cut for that camera's image)


class CropExtractorSettings(BaseSettings):
    """Configuration for GPU-based image cropping."""
    # Padding around the person's box, as a fraction of it, on both axes. Live: it only moves the
    # crop window. One value, not one per axis: the crop is then grown to the output's aspect, which
    # recomputes one axis from the other, so only one padding ever counts — the height for a narrow
    # (standing) box, the width for a wide one (arms out, crouching). RTMPose was trained on boxes
    # padded by 0.25, scale-jittered around that.
    padding:          Field[float] = Field(0.0, min=0.0, max=1.0, step=0.05,
                                           description="Margin added around the box before cropping (fraction; RTMPose uses 0.25)")
    output_width:     Field[int]   = Field(384, access=Field.INIT)
    output_height:    Field[int]   = Field(512, access=Field.INIT)
    max_players:        Field[int]   = Field(4, min=1, max=16, access=Field.INIT)
    enable_prev_crop: Field[bool]  = Field(False, access=Field.INIT, description="Cut the previous frame at the box too, for the optical-flow model")
    interval:         Field[int]   = Field(1, min=1, max=8, description="Frames from one cut to the next: 1 every frame, 2 every other")
