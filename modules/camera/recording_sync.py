"""RecordingSync: the color frame, its depth, the camera's tracklets and the host's poses of one capture, brought
together for the recorder. Completeness first: every capture waits for its parts in the order of capture, as long
as the slowest takes, and a newer capture never ends the wait; the live `CaptureSync` puts latency first instead.

A capture is complete with its color frame and, when depth is recorded, its depth, matched by capture time
(`CaptureStream`). Once a camera has sent tracklets it sends them for every frame, about 33 ms after the frame's
capture on the P800, so a complete capture then waits for its own up to `RecorderSettings.tracklet_wait`;
a camera that never sent any is not waited for. The poses come only for a frame the pose model was asked poses
for: once the app has said for any frame whether it asked (`expect_poses`), a capture waits up to
`RecorderSettings.pose_wait` to know, and for the poses it was asked; a frame nobody asked poses for does not
wait for them. Captures go out in order: one still waiting holds the newer ones behind it. A color frame or a
depth no capture took, and a capture that went out without its tracklets or the poses asked for, are counted and
warned of (`MissedParts`: `depth`, `color`, `tracklets`, `poses`). The submits only take note, on the threads they
come on; `update`, on the recorder's thread, sends out the captures whose wait has ended."""

import time
from collections import defaultdict, deque
from dataclasses import dataclass
from threading import Lock
from typing import Callable, TypeAlias

from .capture_match import CaptureStream, MissedParts
from .chunk import PoseArrays
from .frame import CAPTURE_TOLERANCE, CameraFrame, FrameType
from .tracklet import CameraTracklets

KEPT: int = 16      # frames of each part kept per camera: the depth trails its color by about one, the tracklets by two


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
    video: CameraFrame
    depth: CameraFrame | None
    tracklet_deadline: float        # perf_counter: when it goes out without its tracklets
    pose_deadline: float            # and without its poses


class RecordingSync:
    def __init__(self, with_depth: bool, tracklet_wait: Callable[[], float], pose_wait: Callable[[], float]) -> None:
        """`tracklet_wait`, `pose_wait`: the longest a capture waits for its tracklets and its poses, in ms, read
        when a capture completes."""
        self.with_depth: bool = with_depth
        self._tracklet_wait: Callable[[], float] = tracklet_wait
        self._pose_wait: Callable[[], float] = pose_wait
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

    # ── Out, on the recorder's thread ─────────────────────────────────

    def update(self) -> None:
        """Send out the captures whose wait for their tracklets and poses has ended."""
        self._send_all(force=False)

    def flush(self) -> None:
        """Send out every capture waiting, without the parts that have not come: the recording ends."""
        self._send_all(force=True)

    def _send_all(self, force: bool) -> None:
        with self._emit_lock:
            with self._lock:
                captures: list[RecordedCapture] = []
                for cam_id in list(self._waiting):
                    captures += self._release(cam_id, force)
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
        now: float = time.perf_counter()
        for timestamp, video, depth in complete:
            self.missed.count(cam_id, 'depth', self._videos[cam_id].take_through(timestamp))     # color frames without depth
            if self.with_depth:
                self.missed.count(cam_id, 'color', self._depths[cam_id].take_through(timestamp))
            self._last[cam_id] = timestamp
            self._waiting[cam_id].append(_Waiting(video, depth, now + self._tracklet_wait() / 1000.0, now + self._pose_wait() / 1000.0))
            self.missed.started()

    def _release(self, cam_id: int, force: bool) -> list[RecordedCapture]:
        """The captures at the head of this camera's line whose parts are in or whose wait for them has ended."""
        captures: list[RecordedCapture] = []
        waiting: deque[_Waiting] = self._waiting[cam_id]
        tracklets: CaptureStream[CameraTracklets] = self._tracklets[cam_id]
        poses: CaptureStream[FramePoses] = self._poses[cam_id]
        now: float = time.perf_counter()
        while waiting:
            head: _Waiting = waiting[0]
            timestamp: float = head.video.timestamp
            found_tracklets: CameraTracklets | None = tracklets.at(timestamp)
            tracklets_due: bool = found_tracklets is None and cam_id in self._sends_tracklets and now < head.tracklet_deadline
            asked: bool | None = self._asks[cam_id].at(timestamp) if cam_id in self._asks_poses else False
            found_poses: FramePoses | None = poses.at(timestamp) if asked else None
            poses_due: bool = (asked is None or (asked and found_poses is None)) and now < head.pose_deadline
            if not force and (tracklets_due or poses_due):
                break
            waiting.popleft()
            if found_tracklets is None and cam_id in self._sends_tracklets:
                self.missed.count(cam_id, 'tracklets')
            if found_poses is None and asked is not False:
                self.missed.count(cam_id, 'poses')      # asked, or never said, and none came
            tracklets.take_through(timestamp)
            poses.take_through(timestamp)
            self._asks[cam_id].take_through(timestamp)
            captures.append(RecordedCapture(head.video, head.depth, found_tracklets, found_poses))
        return captures

    def _emit(self, captures: list[RecordedCapture]) -> None:
        for capture in captures:
            for callback in self._callbacks:
                callback(capture)
