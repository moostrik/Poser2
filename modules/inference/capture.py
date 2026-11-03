"""A capture: everything known about one camera's moment of capture, in one piece."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, TypeAlias

from modules.camera import CameraFrame
from modules.pose import FrameDict

from .source import GpuImage

if TYPE_CHECKING:
    from .segmentation import ImageDict


@dataclass(frozen=True)
class Capture:
    """One camera's moment: the color image on the GPU, the depth frame, and the poses and RVM's
    masks found in that image, all captured at `timestamp` (`CameraFrame.timestamp`, within
    `CAPTURE_TOLERANCE`).

    `poses` are the poses of this moment when `poses_synced`, else the newest ones from before it,
    or none: a capture never waits for poses longer than `CaptureSyncSettings.pose_wait`. `masks`
    (per track_id) likewise, with `masks_synced` and `mask_wait`.
    """
    cam_id: int
    timestamp: float
    image: GpuImage
    depth: CameraFrame
    poses: FrameDict = field(default_factory=dict)
    poses_synced: bool = False
    masks: "ImageDict" = field(default_factory=dict)
    masks_synced: bool = False


CaptureCallback: TypeAlias = Callable[[Capture], None]
