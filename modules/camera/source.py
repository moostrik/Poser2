"""The camera contract: what an app and the pipeline see of a camera, whichever device fulfils it."""

from typing import Callable, Protocol, TypeAlias

import numpy as np

from .frame import CameraFrame, FrameType
from .settings import CameraReadings
from .tracklet import CameraTracklets

CameraFrameCallback: TypeAlias = Callable[[CameraFrame], None]
TrackletCallback: TypeAlias = Callable[[CameraTracklets], None]
# A recorded frame: (cam_id, frame_type, image, what was recorded with it, its capture on the play's clock).
ClipFrameCallback: TypeAlias = Callable[[int, FrameType, np.ndarray, object, float | None], None]


class ClipSource(Protocol):
    """Recorded frames for a camera to feed through its device: the recording's player."""

    def add_frame_callback(self, callback: ClipFrameCallback) -> None: ...


class CameraSource(Protocol):
    """One camera: a thread that delivers frames and, when it runs a detector, tracklets; `cam_id` is its index in
    the app and on all it emits, `connected` whether its device is open now."""
    cam_id: int

    @property
    def readings(self) -> CameraReadings: ...

    @property
    def connected(self) -> bool: ...

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def join(self, timeout: float | None = None) -> None: ...
    def add_frame_callback(self, callback: CameraFrameCallback) -> None: ...
    def add_tracklet_callback(self, callback: TrackletCallback) -> None: ...
