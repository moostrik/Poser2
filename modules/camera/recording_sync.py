"""RecordingSync: the color frame, its depth, the camera's tracklets and the host's poses of one capture, brought
together for the recorder. Completeness first: every capture waits for its parts in the order of capture, and a
newer capture never ends the wait; the live `CaptureSync` puts latency first instead.

Each camera is matched apart, under its own lock, as live: one camera's parts never wait for another's. A capture is
complete with its color frame and, when depth is recorded and the camera sends it, its depth, matched by capture time
(`CaptureStream`); a camera that sent no depth while `KEPT` color frames came is a color-only camera, recorded without
depth. Then a capture waits for its tracklets, once the camera has sent any, and for its poses, once the app says per
frame whether the pose model was asked poses for it (`expect_poses`); a frame nobody asked poses for does not wait
for them. No wait is timed: each stream comes in the order of capture, so a part of a later frame says that this
frame's part will not come. Then the capture goes out without it, counted and warned of (`MissedParts`: `tracklets`,
`poses`; and a color frame or a depth no capture took: `depth`, `color`). A camera's captures go out in order: one
still waiting holds the newer ones behind it, at most `KEPT` of them, beyond which the oldest goes out without what it
waits for, for a stream that stopped. The submits only take note, on the threads they come on."""

from collections import deque
from dataclasses import dataclass, field
from threading import Lock
from typing import Callable, TypeAlias

from .capture_match import CaptureStream, MissedParts
from .chunk import PoseArrays
from .frame import CAPTURE_TOLERANCE, CameraFrame, FrameType
from .tracklet import CameraTracklets

KEPT: int = 16      # frames of each part kept, and captures waiting, per camera: 320 ms at 50 fps, three times the slowest pose


@dataclass(frozen=True)
class FramePoses:
    """The host's poses of one camera's frame, as the recorder keeps them: `timestamp` the frame's capture,
    `arrival` the moment the poses were handed to the recorder, `poses` by track_id (`chunk.PoseArrays`)."""
    cam_id: int
    timestamp: float
    poses: PoseArrays
    arrival: float


@dataclass(frozen=True)
class RecordedCapture:
    """One capture as the recorder writes it: the color frame, its depth (None without depth recorded or sent), the
    camera's tracklets of it and the host's poses of it (None when none came)."""
    video: CameraFrame
    depth: CameraFrame | None
    tracklets: CameraTracklets | None
    poses: FramePoses | None = None

    @property
    def nbytes(self) -> int:
        """The bytes of its images: what it holds in a queue."""
        return self.video.image.nbytes + (self.depth.image.nbytes if self.depth is not None else 0)


RecordedCaptureCallback: TypeAlias = Callable[[RecordedCapture], None]


@dataclass
class _Waiting:
    """A complete capture and the parts it got as they came, kept with it so newer parts cannot push them out."""
    video: CameraFrame
    depth: CameraFrame | None
    tracklets: CameraTracklets | None = None
    asked: bool | None = None           # whether poses were asked for its frame; None until the app says
    poses: FramePoses | None = None


@dataclass
class _Camera:
    """One camera's parts, waiting captures and what it has shown it sends, under its own locks."""
    lock: Lock = field(default_factory=Lock)
    emit_lock: Lock = field(default_factory=Lock)       # matching and sending as one step, so its captures leave in order
    videos: CaptureStream[CameraFrame] = field(default_factory=lambda: CaptureStream(KEPT))
    depths: CaptureStream[CameraFrame] = field(default_factory=lambda: CaptureStream(KEPT))
    tracklets: CaptureStream[CameraTracklets] = field(default_factory=lambda: CaptureStream(KEPT))
    asks: CaptureStream[bool] = field(default_factory=lambda: CaptureStream(KEPT))
    poses: CaptureStream[FramePoses] = field(default_factory=lambda: CaptureStream(KEPT))
    last: float = float('-inf')                         # the capture time of its last complete capture
    sends_depth: bool = False
    color_only: bool = False                            # it sent no depth while `KEPT` color frames came
    sends_tracklets: bool = False
    asks_poses: bool = False                            # the app says per frame whether it asked poses
    waiting: deque[_Waiting] = field(default_factory=deque)


def _later(stream: CaptureStream, timestamp: float) -> bool:
    """A part of a frame captured after `timestamp` has come: the stream is past it."""
    newest: float | None = stream.newest()
    return newest is not None and newest > timestamp + CAPTURE_TOLERANCE


class RecordingSync:
    def __init__(self, with_depth: bool) -> None:
        self.with_depth: bool = with_depth
        self._cameras_lock: Lock = Lock()
        self._cameras: dict[int, _Camera] = {}
        self.missed: MissedParts = MissedParts('Recorder')
        self._callbacks: list[RecordedCaptureCallback] = []

    def add_capture_callback(self, callback: RecordedCaptureCallback) -> None:
        if callback not in self._callbacks:
            self._callbacks.append(callback)

    def reset(self) -> None:
        """A new recording: nothing kept, nothing waiting, the counts from zero."""
        with self._cameras_lock:
            self._cameras = {}
        self.missed.reset()

    def _camera(self, cam_id: int) -> _Camera:
        with self._cameras_lock:
            camera: _Camera | None = self._cameras.get(cam_id)
            if camera is None:
                camera = self._cameras[cam_id] = _Camera()
            return camera

    @property
    def waiting(self) -> int:
        """The captures complete and waiting for their tracklets or poses, over every camera."""
        with self._cameras_lock:
            cameras: list[_Camera] = list(self._cameras.values())
        total: int = 0
        for camera in cameras:
            with camera.lock:
                total += len(camera.waiting)
        return total

    # ── Inputs ────────────────────────────────────────────────────────

    def submit_frame(self, frame: CameraFrame) -> None:
        if frame.kind == FrameType.DEPTH and not self.with_depth:
            return
        camera: _Camera = self._camera(frame.cam_id)
        with camera.emit_lock:
            with camera.lock:
                if frame.kind == FrameType.VIDEO:
                    if camera.videos.add(frame.timestamp, frame):
                        self.missed.count(frame.cam_id, 'depth')         # a color frame pushed out unmatched
                else:
                    camera.sends_depth = True
                    if camera.depths.add(frame.timestamp, frame):
                        self.missed.count(frame.cam_id, 'color')         # a depth pushed out unmatched
                self._complete(frame.cam_id, camera)
                captures: list[RecordedCapture] = self._release(frame.cam_id, camera, force=False)
            self._emit(captures)

    def submit_tracklets(self, cam_tracklets: CameraTracklets) -> None:
        camera: _Camera = self._camera(cam_tracklets.cam_id)
        with camera.emit_lock:
            with camera.lock:
                camera.tracklets.add(cam_tracklets.timestamp, cam_tracklets)
                camera.sends_tracklets = True
                captures: list[RecordedCapture] = self._release(cam_tracklets.cam_id, camera, force=False)
            self._emit(captures)

    def expect_poses(self, cam_id: int, timestamp: float, asked: bool) -> None:
        """Whether the pose model was asked poses for the camera's frame captured at `timestamp`."""
        camera: _Camera = self._camera(cam_id)
        with camera.emit_lock:
            with camera.lock:
                camera.asks.add(timestamp, asked)
                camera.asks_poses = True
                captures: list[RecordedCapture] = self._release(cam_id, camera, force=False)
            self._emit(captures)

    def submit_poses(self, frame_poses: FramePoses) -> None:
        camera: _Camera = self._camera(frame_poses.cam_id)
        with camera.emit_lock:
            with camera.lock:
                camera.poses.add(frame_poses.timestamp, frame_poses)
                captures: list[RecordedCapture] = self._release(frame_poses.cam_id, camera, force=False)
            self._emit(captures)

    def flush(self) -> None:
        """Send out every capture waiting, without the parts that have not come: the recording ends."""
        with self._cameras_lock:
            cameras: dict[int, _Camera] = dict(self._cameras)
        for cam_id, camera in cameras.items():
            with camera.emit_lock:
                with camera.lock:
                    captures: list[RecordedCapture] = self._release(cam_id, camera, force=True)
                self._emit(captures)

    # ── Matching, under the camera's lock ─────────────────────────────

    def _complete(self, cam_id: int, camera: _Camera) -> None:
        """The camera's complete captures newer than its last, oldest first, into its waiting line."""
        if self.with_depth and not camera.sends_depth and not camera.color_only and len(camera.videos) >= KEPT:
            camera.color_only = True        # no depth while `KEPT` color frames came: recorded without depth
        complete: list[tuple[float, CameraFrame, CameraFrame | None]] = []
        if self.with_depth and not camera.color_only:
            for depth_time, depth in camera.depths:
                if depth_time > camera.last + CAPTURE_TOLERANCE:
                    video: CameraFrame | None = camera.videos.at(depth_time)
                    if video is not None:
                        complete.append((video.timestamp, video, depth))
        else:
            complete = [(video_time, video, None) for video_time, video in camera.videos if video_time > camera.last + CAPTURE_TOLERANCE]
        for timestamp, video, depth in complete:
            self.missed.count(cam_id, 'depth', camera.videos.take_through(timestamp))     # color frames without depth
            if self.with_depth:
                self.missed.count(cam_id, 'color', camera.depths.take_through(timestamp))
            camera.last = timestamp
            camera.waiting.append(_Waiting(video, depth))
            self.missed.started(cam_id)

    def _attach(self, camera: _Camera) -> None:
        """The parts come in the waiting captures' streams kept with them, before newer ones can push them out."""
        for waiting in camera.waiting:
            timestamp: float = waiting.video.timestamp
            if waiting.tracklets is None:
                waiting.tracklets = camera.tracklets.at(timestamp)
            if waiting.asked is None:
                waiting.asked = camera.asks.at(timestamp) if camera.asks_poses else False
            if waiting.asked and waiting.poses is None:
                waiting.poses = camera.poses.at(timestamp)

    def _release(self, cam_id: int, camera: _Camera, force: bool) -> list[RecordedCapture]:
        """The captures at the head of the camera's line whose parts are in, or whose streams are past them."""
        self._attach(camera)
        captures: list[RecordedCapture] = []
        waiting: deque[_Waiting] = camera.waiting
        while waiting:
            head: _Waiting = waiting[0]
            timestamp: float = head.video.timestamp
            found_tracklets: CameraTracklets | None = head.tracklets
            tracklets_due: bool = found_tracklets is None and camera.sends_tracklets and not _later(camera.tracklets, timestamp)
            asked: bool | None = head.asked
            if asked is None and _later(camera.asks, timestamp):
                asked = False           # the stream is past it without a word: no image was posed
            found_poses: FramePoses | None = head.poses if asked else None
            poses_due: bool = asked is None or (asked and found_poses is None and not _later(camera.poses, timestamp))
            if not force and len(waiting) <= KEPT and (tracklets_due or poses_due):
                break
            waiting.popleft()
            if found_tracklets is None and camera.sends_tracklets:
                self.missed.count(cam_id, 'tracklets')
            if found_poses is None and asked is not False:
                self.missed.count(cam_id, 'poses')      # asked, or never said, and none came
            camera.tracklets.take_through(timestamp)
            camera.poses.take_through(timestamp)
            camera.asks.take_through(timestamp)
            captures.append(RecordedCapture(head.video, head.depth, found_tracklets, found_poses))
        return captures

    def _emit(self, captures: list[RecordedCapture]) -> None:
        for capture in captures:
            for callback in self._callbacks:
                callback(capture)
