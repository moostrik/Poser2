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
    """One camera: a thread that delivers frames and, when it runs a detector, tracklets.

    `id` is the camera's index in the app, `cam_id` on everything it emits. Callbacks are
    registered before `start()` and run on the backend's own thread in registration order, so
    they must be short and thread-safe. A frame or tracklet message is stamped once at hand-over
    with the moment of its capture, or its arrival where the backend cannot know that, on
    `time.perf_counter()` (`CameraFrame.timestamp`, `CameraTracklets.timestamp`), and, where the
    two differ, with its arrival (`arrival`). The tracklet callback gets every message
    (`CameraTracklets`), an empty one included: no person is a tick too; a backend without a detector never calls it. `connected` says whether the device is open now: a backend that keeps looking
    for its device (Vimba) is started without one and connects when it is there.
    """
    id: int

    @property
    def readings(self) -> CameraReadings: ...

    @property
    def connected(self) -> bool: ...

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def join(self, timeout: float | None = None) -> None: ...
    def add_frame_callback(self, callback: CameraFrameCallback) -> None: ...
    def add_tracklet_callback(self, callback: TrackletCallback) -> None: ...
