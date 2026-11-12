"""RecordingSync: the color frame, its depth and the camera's tracklets of one capture, brought together for the
recorder. Completeness first: every capture waits for its parts in the order of capture, as long as the slowest
takes, and a newer capture never ends the wait; the live `CaptureSync` puts latency first instead.

A capture is complete with its color frame and, when depth is recorded, its depth, matched by capture time
(`CaptureStream`). Once a camera has sent tracklets it sends them for every frame, about 33 ms after the frame's
capture on the P800, so a complete capture then waits for its own up to `RecorderSettings.tracklet_wait`;
a camera that never sent any is not waited for. Captures go out in order: one still waiting holds the newer
ones behind it. A color frame or a depth no capture took, and a capture that went out without its tracklets,
are counted and warned of (`MissedParts`: `depth`, `color`, `tracklets`). The submits only take note, on the
camera's threads; `update`, on the recorder's thread, sends out the captures whose wait has ended."""

import time
from collections import defaultdict, deque
from dataclasses import dataclass
from threading import Lock
from typing import Callable, TypeAlias

from .capture_match import CaptureStream, MissedParts
from .frame import CAPTURE_TOLERANCE, CameraFrame, FrameType
from .tracklet import CameraTracklets

KEPT: int = 16      # frames of each part kept per camera: the depth trails its color by about one, the tracklets by two


@dataclass(frozen=True)
class RecordedCapture:
    """One capture as the recorder writes it: the color frame, its depth (None without depth recorded) and the
    camera's tracklets of it (None when none came)."""
    video: CameraFrame
    depth: CameraFrame | None
    tracklets: CameraTracklets | None


RecordedCaptureCallback: TypeAlias = Callable[[RecordedCapture], None]


@dataclass
class _Waiting:
    video: CameraFrame
    depth: CameraFrame | None
    deadline: float         # perf_counter: when it goes out without its tracklets


class RecordingSync:
    def __init__(self, with_depth: bool, tracklet_wait: Callable[[], float]) -> None:
        """`tracklet_wait`: the longest a capture waits for its tracklets, in ms, read when a capture completes."""
        self.with_depth: bool = with_depth
        self._tracklet_wait: Callable[[], float] = tracklet_wait
        self._lock: Lock = Lock()
        self._emit_lock: Lock = Lock()          # matching and sending as one step, so captures leave in order
        self._videos: dict[int, CaptureStream[CameraFrame]] = defaultdict(lambda: CaptureStream(KEPT))
        self._depths: dict[int, CaptureStream[CameraFrame]] = defaultdict(lambda: CaptureStream(KEPT))
        self._tracklets: dict[int, CaptureStream[CameraTracklets]] = defaultdict(lambda: CaptureStream(KEPT))
        self._last: dict[int, float] = {}                       # the capture time of each camera's last complete capture
        self._sends_tracklets: set[int] = set()                 # the cameras that have sent tracklets
        self._waiting: dict[int, deque[_Waiting]] = defaultdict(deque)
        self.missed: MissedParts = MissedParts('Recorder')
        self._callbacks: list[RecordedCaptureCallback] = []

    def add_capture_callback(self, callback: RecordedCaptureCallback) -> None:
        if callback not in self._callbacks:
            self._callbacks.append(callback)

    def reset(self) -> None:
        """A new recording: nothing kept, nothing waiting, the counts from zero."""
        with self._emit_lock, self._lock:
            for streams in (self._videos, self._depths, self._tracklets):
                streams.clear()
            self._last.clear()
            self._sends_tracklets.clear()
            self._waiting.clear()
            self.missed.reset()

    # ── Inputs, on the camera's threads ───────────────────────────────

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

    # ── Out, on the recorder's thread ─────────────────────────────────

    def update(self) -> None:
        """Send out the captures whose wait for their tracklets has ended."""
        self._send_all(force=False)

    def flush(self) -> None:
        """Send out every capture waiting, without the tracklets that have not come: the recording ends."""
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
        for timestamp, video, depth in complete:
            self.missed.count(cam_id, 'depth', self._videos[cam_id].take_through(timestamp))     # color frames without depth
            if self.with_depth:
                self.missed.count(cam_id, 'color', self._depths[cam_id].take_through(timestamp))
            self._last[cam_id] = timestamp
            wait: float = self._tracklet_wait() / 1000.0 if cam_id in self._sends_tracklets else 0.0
            self._waiting[cam_id].append(_Waiting(video, depth, time.perf_counter() + wait))
            self.missed.started()

    def _release(self, cam_id: int, force: bool) -> list[RecordedCapture]:
        """The captures at the head of this camera's line whose tracklets are in or whose wait has ended."""
        captures: list[RecordedCapture] = []
        waiting: deque[_Waiting] = self._waiting[cam_id]
        tracklets: CaptureStream[CameraTracklets] = self._tracklets[cam_id]
        now: float = time.perf_counter()
        while waiting:
            head: _Waiting = waiting[0]
            found: CameraTracklets | None = tracklets.at(head.video.timestamp)
            if found is None and not force and now < head.deadline:
                break
            waiting.popleft()
            if found is None and cam_id in self._sends_tracklets:
                self.missed.count(cam_id, 'tracklets')
            tracklets.take_through(head.video.timestamp)
            captures.append(RecordedCapture(head.video, head.depth, found))
        return captures

    def _emit(self, captures: list[RecordedCapture]) -> None:
        for capture in captures:
            for callback in self._callbacks:
                callback(capture)
