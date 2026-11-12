"""RecordingSync: the color frame, its depth, the camera's tracklets and the host's poses of one capture, brought
together for the recorder. Completeness first: every capture waits for its parts in the order of capture, and a
newer capture never ends the wait; the live `CaptureSync` puts latency first instead.

A capture is complete with its color frame and, when depth is recorded, its depth, matched by capture time
(`CaptureStream`). Then it waits for its tracklets, once the camera has sent any, and for its poses, once the app
says per frame whether the pose model was asked poses for it (`expect_poses`); a frame nobody asked poses for does
not wait for them. No wait is timed: each stream comes in the order of capture, so a part of a later frame says
that this frame's part will not come. Then the capture goes out without it, counted and warned of (`MissedParts`:
`tracklets`, `poses`; and a color frame or a depth no capture took: `depth`, `color`). Captures go out in order:
one still waiting holds the newer ones behind it, at most `KEPT` of them, beyond which the oldest goes out without
what it waits for, for a stream that stopped. The submits only take note, on the threads they come on."""

from collections import defaultdict, deque
from dataclasses import dataclass
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
    """One capture as the recorder writes it: the color frame, its depth (None without depth recorded), the
    camera's tracklets of it and the host's poses of it (None when none came)."""
    video: CameraFrame
    depth: CameraFrame | None
    tracklets: CameraTracklets | None
    poses: FramePoses | None = None


RecordedCaptureCallback: TypeAlias = Callable[[RecordedCapture], None]


@dataclass
class _Waiting:
    """A complete capture and the parts it got as they came, kept with it so newer parts cannot push them out."""
    video: CameraFrame
    depth: CameraFrame | None
    tracklets: CameraTracklets | None = None
    asked: bool | None = None           # whether poses were asked for its frame; None until the app says
    poses: FramePoses | None = None


def _later(stream: CaptureStream, timestamp: float) -> bool:
    """A part of a frame captured after `timestamp` has come: the stream is past it."""
    newest: float | None = stream.newest()
    return newest is not None and newest > timestamp + CAPTURE_TOLERANCE


class RecordingSync:
    def __init__(self, with_depth: bool) -> None:
        self.with_depth: bool = with_depth
        self._lock: Lock = Lock()
        self._emit_lock: Lock = Lock()          # matching and sending as one step, so captures leave in order
        self._videos: dict[int, CaptureStream[CameraFrame]] = defaultdict(lambda: CaptureStream(KEPT))
        self._depths: dict[int, CaptureStream[CameraFrame]] = defaultdict(lambda: CaptureStream(KEPT))
        self._tracklets: dict[int, CaptureStream[CameraTracklets]] = defaultdict(lambda: CaptureStream(KEPT))
        self._asks: dict[int, CaptureStream[bool]] = defaultdict(lambda: CaptureStream(KEPT))
        self._poses: dict[int, CaptureStream[FramePoses]] = defaultdict(lambda: CaptureStream(KEPT))
        self._last: dict[int, float] = {}                       # the capture time of each camera's last complete capture
        self._sends_tracklets: set[int] = set()                 # the cameras that have sent tracklets
        self._asks_poses: set[int] = set()                      # the cameras the app says per frame whether it asked poses
        self._waiting: dict[int, deque[_Waiting]] = defaultdict(deque)
        self.missed: MissedParts = MissedParts('Recorder')
        self._callbacks: list[RecordedCaptureCallback] = []

    def add_capture_callback(self, callback: RecordedCaptureCallback) -> None:
        if callback not in self._callbacks:
            self._callbacks.append(callback)

    def reset(self) -> None:
        """A new recording: nothing kept, nothing waiting, the counts from zero."""
        with self._emit_lock, self._lock:
            for streams in (self._videos, self._depths, self._tracklets, self._asks, self._poses):
                streams.clear()
            self._last.clear()
            self._sends_tracklets.clear()
            self._asks_poses.clear()
            self._waiting.clear()
            self.missed.reset()

    @property
    def waiting(self) -> int:
        """The captures complete and waiting for their tracklets or poses, over every camera."""
        with self._lock:
            return sum(len(waiting) for waiting in self._waiting.values())

    # ── Inputs ────────────────────────────────────────────────────────

    def submit_frame(self, frame: CameraFrame) -> None:
        if frame.kind == FrameType.DEPTH and not self.with_depth:
            return
        with self._emit_lock:
            with self._lock:
                if frame.kind == FrameType.VIDEO:
                    if self._videos[frame.cam_id].add(frame.timestamp, frame):
                        self.missed.count(frame.cam_id, 'depth')         # a color frame pushed out unmatched
                else:
                    if self._depths[frame.cam_id].add(frame.timestamp, frame):
                        self.missed.count(frame.cam_id, 'color')         # a depth pushed out unmatched
                self._complete(frame.cam_id)
                captures: list[RecordedCapture] = self._release(frame.cam_id, force=False)
            self._emit(captures)

    def submit_tracklets(self, cam_tracklets: CameraTracklets) -> None:
        with self._emit_lock:
            with self._lock:
                self._tracklets[cam_tracklets.cam_id].add(cam_tracklets.timestamp, cam_tracklets)
                self._sends_tracklets.add(cam_tracklets.cam_id)
                captures: list[RecordedCapture] = self._release(cam_tracklets.cam_id, force=False)
            self._emit(captures)

    def expect_poses(self, cam_id: int, timestamp: float, asked: bool) -> None:
        """Whether the pose model was asked poses for the camera's frame captured at `timestamp`."""
        with self._emit_lock:
            with self._lock:
                self._asks[cam_id].add(timestamp, asked)
                self._asks_poses.add(cam_id)
                captures: list[RecordedCapture] = self._release(cam_id, force=False)
            self._emit(captures)

    def submit_poses(self, frame_poses: FramePoses) -> None:
        with self._emit_lock:
            with self._lock:
                self._poses[frame_poses.cam_id].add(frame_poses.timestamp, frame_poses)
                captures: list[RecordedCapture] = self._release(frame_poses.cam_id, force=False)
            self._emit(captures)

    def flush(self) -> None:
        """Send out every capture waiting, without the parts that have not come: the recording ends."""
        with self._emit_lock:
            with self._lock:
                captures: list[RecordedCapture] = []
                for cam_id in list(self._waiting):
                    captures += self._release(cam_id, force=True)
            self._emit(captures)

    # ── Matching, under the lock ──────────────────────────────────────

    def _complete(self, cam_id: int) -> None:
        """This camera's complete captures newer than its last, oldest first, into its waiting line."""
        last: float = self._last.get(cam_id, float('-inf'))
        complete: list[tuple[float, CameraFrame, CameraFrame | None]] = []
        if self.with_depth:
            for depth_time, depth in self._depths[cam_id]:
                if depth_time > last + CAPTURE_TOLERANCE:
                    video: CameraFrame | None = self._videos[cam_id].at(depth_time)
                    if video is not None:
                        complete.append((video.timestamp, video, depth))
        else:
            complete = [(video_time, video, None) for video_time, video in self._videos[cam_id] if video_time > last + CAPTURE_TOLERANCE]
        for timestamp, video, depth in complete:
            self.missed.count(cam_id, 'depth', self._videos[cam_id].take_through(timestamp))     # color frames without depth
            if self.with_depth:
                self.missed.count(cam_id, 'color', self._depths[cam_id].take_through(timestamp))
            self._last[cam_id] = timestamp
            self._waiting[cam_id].append(_Waiting(video, depth))
            self.missed.started()

    def _attach(self, cam_id: int) -> None:
        """The parts come in the waiting captures' streams kept with them, before newer ones can push them out."""
        tracklets: CaptureStream[CameraTracklets] = self._tracklets[cam_id]
        asks: CaptureStream[bool] = self._asks[cam_id]
        poses: CaptureStream[FramePoses] = self._poses[cam_id]
        for waiting in self._waiting[cam_id]:
            timestamp: float = waiting.video.timestamp
            if waiting.tracklets is None:
                waiting.tracklets = tracklets.at(timestamp)
            if waiting.asked is None:
                waiting.asked = asks.at(timestamp) if cam_id in self._asks_poses else False
            if waiting.asked and waiting.poses is None:
                waiting.poses = poses.at(timestamp)

    def _release(self, cam_id: int, force: bool) -> list[RecordedCapture]:
        """The captures at the head of this camera's line whose parts are in, or whose streams are past them."""
        self._attach(cam_id)
        captures: list[RecordedCapture] = []
        waiting: deque[_Waiting] = self._waiting[cam_id]
        tracklets: CaptureStream[CameraTracklets] = self._tracklets[cam_id]
        asks: CaptureStream[bool] = self._asks[cam_id]
        poses: CaptureStream[FramePoses] = self._poses[cam_id]
        while waiting:
            head: _Waiting = waiting[0]
            timestamp: float = head.video.timestamp
            found_tracklets: CameraTracklets | None = head.tracklets
            tracklets_due: bool = found_tracklets is None and cam_id in self._sends_tracklets and not _later(tracklets, timestamp)
            asked: bool | None = head.asked
            if asked is None and _later(asks, timestamp):
                asked = False           # the stream is past it without a word: no image was posed
            found_poses: FramePoses | None = head.poses if asked else None
            poses_due: bool = asked is None or (asked and found_poses is None and not _later(poses, timestamp))
            if not force and len(waiting) <= KEPT and (tracklets_due or poses_due):
                break
            waiting.popleft()
            if found_tracklets is None and cam_id in self._sends_tracklets:
                self.missed.count(cam_id, 'tracklets')
            if found_poses is None and asked is not False:
                self.missed.count(cam_id, 'poses')      # asked, or never said, and none came
            tracklets.take_through(timestamp)
            poses.take_through(timestamp)
            asks.take_through(timestamp)
            captures.append(RecordedCapture(head.video, head.depth, found_tracklets, found_poses))
        return captures

    def _emit(self, captures: list[RecordedCapture]) -> None:
        for capture in captures:
            for callback in self._callbacks:
                callback(capture)
