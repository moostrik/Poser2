"""RecordingSync: the color frame, its depth, the camera's tracklets and the host's poses of one capture, brought
together for the recorder. Completeness first: every capture waits for its parts in the order of capture, and a
newer capture never ends the wait; the live `InferenceFrameSync` puts latency first instead.

Each camera is matched apart, as live: one camera's parts never wait for another's. A capture is complete with its
color frame and, when depth is recorded and the camera sends it, its depth, matched by capture time
(`CaptureStream`); a camera that sent no depth while `KEPT` color frames came is a color-only camera, recorded without
depth. Then a capture waits for its tracklets, once the camera has sent any, and for its poses, once the app says per
frame whether the pose model was asked poses for it (`expect_poses`); a frame nobody asked poses for does not wait
for them. No wait is timed: each stream comes in the order of capture, so a part of a later frame says that this
frame's part will not come. Then the capture goes out without it, counted and warned of (`MissedParts`: `tracklets`,
`poses`; and a color frame or a depth no capture took: `depth`, `color`). A camera's captures go out in order: one
still waiting holds the newer ones behind it, at most `KEPT` of them, beyond which the oldest goes out without what it
waits for, for a stream that stopped.

The submits only queue what they take, on the threads they come on, in one inbox that is not bounded: a recording
needs every part. The sync's own thread matches the parts in the order they came and hands the captures on. A part
before `start` is an error; one after `stop` is dropped."""

import logging
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum, auto
from threading import Condition, Event, Lock, Thread
from typing import Any, Callable, TypeAlias

import numpy as np

from modules.camera import CAPTURE_TOLERANCE, CameraFrame, CameraTracklets, CaptureStream, FrameType, MissedParts
from modules.pose import BBox, FrameDict, Points2D, has_points
from modules.utils import Broadcast, Rect

from .chunk import PoseArrays

logger = logging.getLogger(__name__)

KEPT: int = 16      # frames of each part kept, and captures waiting, per camera: 320 ms at 50 fps, three times the slowest pose


@dataclass(frozen=True)
class FramePoses:
    """The host's poses of one camera's frame, as the recorder keeps them: `timestamp` the frame's capture,
    `arrival` the moment the poses were handed to the recorder, `poses` by track_id (`chunk.PoseArrays`)."""
    cam_id: int
    timestamp: float
    poses: PoseArrays
    arrival: float


def pose_arrays(frames: FrameDict) -> dict[int, tuple[float, PoseArrays]]:
    """The poses with points, per camera: their frame's capture time and each pose's box and points."""
    by_cam: dict[int, tuple[float, PoseArrays]] = {}
    for track_id, frame in frames.items():
        if not has_points(frame):
            continue
        points: Points2D = frame[Points2D]
        rect: Rect = frame[BBox].to_rect()
        box: np.ndarray = np.array([rect.x, rect.y, rect.width, rect.height], dtype=np.float32)
        arrays: np.ndarray = np.column_stack((points.values, points.scores)).astype(np.float32)
        timestamp, cam_poses = by_cam.get(frame.cam_id, (frame.timestamp, {}))
        cam_poses[track_id] = (box, arrays)
        by_cam[frame.cam_id] = (max(timestamp, frame.timestamp), cam_poses)
    return by_cam


def frame_poses(frames: FrameDict, arrival: float) -> list[FramePoses]:
    """The RAW stage's poses as the recorder keeps them, one per camera, handed over at `arrival`."""
    return [FramePoses(cam_id, timestamp, poses, arrival) for cam_id, (timestamp, poses) in pose_arrays(frames).items()]


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
        depth: int = 0
        if self.depth is not None:
            depth = self.depth.image.nbytes + (self.depth.confidence.nbytes if self.depth.confidence is not None else 0)
        return self.video.image.nbytes + depth


RecordedCaptureCallback: TypeAlias = Callable[[RecordedCapture], None]


class _Part(IntEnum):
    """What waits in the sync's inbox."""
    FRAME = 0
    TRACKLETS = auto()
    ASKED = auto()      # (cam_id, timestamp, asked): whether poses were asked for a frame
    POSES = auto()      # (the app's poses, arrival), packed on the sync's thread
    FLUSH = auto()      # an Event, set once every capture waiting has gone out
    RESET = auto()      # an Event, set once nothing is kept
    SETTLE = auto()     # an Event, set once every part queued before it is matched


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
    """One camera's parts, waiting captures and what it has shown it sends; its lock guards what `waiting` reads."""
    lock: Lock = field(default_factory=Lock)
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


class RecordingSync(Thread):
    def __init__(self, with_depth: bool) -> None:
        super().__init__(daemon=True, name='RecordingSync')
        self.with_depth: bool = with_depth
        self._cameras_lock: Lock = Lock()
        self._cameras: dict[int, _Camera] = {}
        self._expected: dict[int, tuple[bool, bool]] = {}      # each camera's parts said ahead: (tracklets, poses)
        self.missed: MissedParts = MissedParts('Recorder')
        self._callbacks: Broadcast[[RecordedCapture]] = Broadcast(name='RecordingSync capture')
        self._inbox: deque[tuple[_Part, Any]] = deque()
        self._wake: Condition = Condition()
        self._running: bool = False
        self._stopped: bool = False

    def add_capture_callback(self, callback: RecordedCaptureCallback) -> None:
        self._callbacks.add_callback(callback)

    # ── Lifecycle ─────────────────────────────────────────────────────

    def start(self) -> None:
        with self._wake:
            self._running = True
        super().start()

    def stop(self) -> None:
        """The thread ends; what still waits in the inbox is dropped and logged."""
        with self._wake:
            self._running = False
            self._stopped = True
            dropped: list[tuple[_Part, Any]] = list(self._inbox)
            self._inbox.clear()
            self._wake.notify()
        for part, value in dropped:
            if part in (_Part.FLUSH, _Part.RESET, _Part.SETTLE):
                value.set()         # nobody waits forever on a sync that stopped
        if dropped:
            logger.info("RecordingSync: %d parts dropped at stop", len(dropped))
        if self.is_alive():
            self.join(timeout=2.0)

    def run(self) -> None:
        while True:
            with self._wake:
                while self._running and not self._inbox:
                    self._wake.wait()
                if not self._running:
                    return
                part, value = self._inbox.popleft()
            try:
                self._take(part, value)
            except Exception:
                logger.exception("RecordingSync: matching a %s failed", part.name.lower())
                if part in (_Part.FLUSH, _Part.RESET, _Part.SETTLE):
                    value.set()

    def _queue(self, part: _Part, value: Any) -> None:
        """Into the inbox. Before `start` an error; after `stop` dropped (a wait on it returns at once)."""
        with self._wake:
            if not self._running:
                if self._stopped:
                    if part in (_Part.FLUSH, _Part.RESET, _Part.SETTLE):
                        value.set()
                    return
                raise RuntimeError("RecordingSync: a part before start")
            self._inbox.append((part, value))
            self._wake.notify()

    def _done(self, part: _Part) -> None:
        """Queue `part` and return once it is taken."""
        done: Event = Event()
        self._queue(part, done)
        done.wait()

    def reset(self) -> None:
        """A new recording: nothing kept, nothing waiting, the counts from zero."""
        self._done(_Part.RESET)

    def settle(self) -> None:
        """Return once every part submitted before is matched."""
        self._done(_Part.SETTLE)

    def expect_parts(self, cam_id: int, tracklets: bool, poses: bool) -> None:
        """Whether a camera sends tracklets and the app says per image whether it asked poses, kept across
        recordings: its first capture then waits for them as every later one does."""
        with self._cameras_lock:
            self._expected[cam_id] = (tracklets, poses)

    def _camera(self, cam_id: int) -> _Camera:
        with self._cameras_lock:
            camera: _Camera | None = self._cameras.get(cam_id)
            if camera is None:
                tracklets, poses = self._expected.get(cam_id, (False, False))
                camera = self._cameras[cam_id] = _Camera(sends_tracklets=tracklets, asks_poses=poses)
            return camera

    @property
    def waiting(self) -> int:
        """The parts queued and not matched yet, and the captures complete and waiting for their tracklets or
        poses, over every camera."""
        with self._wake:
            total: int = len(self._inbox)
        with self._cameras_lock:
            cameras: list[_Camera] = list(self._cameras.values())
        for camera in cameras:
            with camera.lock:
                total += len(camera.waiting)
        return total

    # ── Inputs: each queues and returns ───────────────────────────────

    def submit_frame(self, frame: CameraFrame) -> None:
        if frame.kind == FrameType.DEPTH and not self.with_depth:
            return
        self._queue(_Part.FRAME, frame)

    def submit_tracklets(self, cam_tracklets: CameraTracklets) -> None:
        self._queue(_Part.TRACKLETS, cam_tracklets)

    def expect_poses(self, cam_id: int, timestamp: float, asked: bool) -> None:
        """Whether the pose model was asked poses for the camera's frame captured at `timestamp`."""
        self._queue(_Part.ASKED, (cam_id, timestamp, asked))

    def submit_poses(self, poses: FrameDict, arrival: float) -> None:
        """The RAW stage's poses as they came out, handed over at `arrival`; packed into `FramePoses` on the sync's
        thread (`frame_poses`)."""
        self._queue(_Part.POSES, (poses, arrival))

    def flush(self) -> None:
        """Send out every capture waiting, without the parts that have not come: the recording ends. Returns once
        they have gone out."""
        self._done(_Part.FLUSH)

    # ── Matching, on the sync's thread ────────────────────────────────

    def _take(self, part: _Part, value: Any) -> None:
        if part == _Part.RESET:
            with self._cameras_lock:
                self._cameras = {}
            self.missed.reset()
            value.set()
            return
        if part == _Part.SETTLE:
            value.set()
            return
        if part == _Part.FLUSH:
            with self._cameras_lock:
                cameras: dict[int, _Camera] = dict(self._cameras)
            for flushed_id, flushed in cameras.items():
                with flushed.lock:
                    released: list[RecordedCapture] = self._release(flushed_id, flushed, force=True)
                self._emit(released)
            value.set()
            return
        if part == _Part.POSES:
            for packed in frame_poses(*value):
                self._take_part(_Part.POSES, packed)
            return
        self._take_part(part, value)

    def _take_part(self, part: _Part, value: Any) -> None:
        """One camera's part: a frame, its tracklets, whether poses were asked, or its packed poses."""
        cam_id: int = value[0] if part == _Part.ASKED else value.cam_id
        camera: _Camera = self._camera(cam_id)
        with camera.lock:
            if part == _Part.FRAME:
                frame: CameraFrame = value
                if frame.kind == FrameType.VIDEO:
                    if camera.videos.add(frame.timestamp, frame):
                        self.missed.count(cam_id, 'depth')          # a color frame pushed out unmatched
                else:
                    camera.sends_depth = True
                    if camera.depths.add(frame.timestamp, frame):
                        self.missed.count(cam_id, 'color')          # a depth pushed out unmatched
                self._complete(cam_id, camera)
            elif part == _Part.TRACKLETS:
                camera.tracklets.add(value.timestamp, value)
                camera.sends_tracklets = True
            elif part == _Part.ASKED:
                _, timestamp, asked = value
                camera.asks.add(timestamp, asked)
                camera.asks_poses = True
            else:
                camera.poses.add(value.timestamp, value)
            captures: list[RecordedCapture] = self._release(cam_id, camera, force=False)
        self._emit(captures)

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
            self._callbacks(capture)
