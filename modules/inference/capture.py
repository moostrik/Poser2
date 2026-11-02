"""A capture: everything known about one camera's moment of capture, in one piece."""

from dataclasses import dataclass, field
from typing import Callable, TypeAlias

from modules.camera import CameraFrame
from modules.pose import FrameDict

from .source import GpuImage


@dataclass(frozen=True)
class Capture:
    """One camera's moment: the color image on the GPU, the depth frame and the poses found in that
    image, all captured at `timestamp` (`CameraFrame.timestamp`, within `CAPTURE_TOLERANCE`).

    `poses` are the poses of this moment when `poses_synced`, else the newest ones from before it,
    or none: a capture never waits for poses longer than `CaptureSyncSettings.pose_wait`.
    """
    cam_id: int
    timestamp: float
    image: GpuImage
    depth: CameraFrame
    poses: FrameDict = field(default_factory=dict)
    poses_synced: bool = False


CaptureCallback: TypeAlias = Callable[[Capture], None]
