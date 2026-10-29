"""The camera contract: what an app and the pipeline see of a camera, whichever device fulfils it."""

from typing import Callable, Protocol, TypeAlias

from .frame import CameraFrame
from .settings import CameraReadings
from .tracklet import CameraTracklet

FrameCallback: TypeAlias = Callable[[CameraFrame], None]
TrackletCallback: TypeAlias = Callable[[int, list[CameraTracklet], float], None]


class CameraSource(Protocol):
    """One camera: a thread that delivers frames and, when it runs a detector, tracklets.

    `id` is the camera's index in the app, `cam_id` on everything it emits. Callbacks are
    registered before `start()` and run on the backend's own thread in registration order, so
    they must be short and thread-safe. A frame or tracklet message is stamped once with the host
    time at hand-over, on `time.perf_counter()` (`CameraFrame.time`, `CameraTracklet.time`). The
    tracklet callback gets every message with its time, an empty list included: no person is a
    tick too; a backend without a detector never calls it. `connected` says whether the device is open now: a backend that keeps looking
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
    def add_frame_callback(self, callback: FrameCallback) -> None: ...
    def add_tracklet_callback(self, callback: TrackletCallback) -> None: ...
