"""CaptureSync: the color image, the depth frame and the poses of one moment of capture, brought together.

The three arrive on their own threads at their own times: the image as it is uploaded (the camera's
video thread), the depth frame about 13 ms later (its own thread), the poses out of a pipeline stage
(the model's thread), measured to arrive with the depth. Each is kept a few frames per camera and
matched by capture time within `CAPTURE_TOLERANCE`. A capture is complete with its image and its
depth; while a player is in view it waits up to `pose_wait` for its own poses, else it takes the
newest poses from before it, so a missing pose never holds it longer. The wait runs on this sync's
own thread: no submit blocks, and a capture goes out on whichever thread completes it.
"""

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from threading import Condition, Lock, Thread

from modules.camera import CAPTURE_TOLERANCE, CameraFrame, FrameType
from modules.pose import FrameDict
from modules.settings import BaseSettings, Field

from .capture import Capture, CaptureCallback
from .source import GpuImage

logger = logging.getLogger(__name__)

KEPT: int = 8       # frames of each stream kept per camera: depth trails its image by about one


class CaptureSyncSettings(BaseSettings):
    """Configuration for CaptureSync."""
    pose_wait: Field[float] = Field(10.0, min=0.0, max=50.0, step=0.5, description="Longest a capture waits for its own poses (ms)")
    misses:    Field[int]   = Field(0, access=Field.READ, description="Depth frames that found no image")


@dataclass
class _Waiting:
    """A capture whose image and depth are in, waiting for its poses until `deadline` (perf_counter)."""
    image: GpuImage
    depth: CameraFrame
    deadline: float


class CaptureSync(Thread):
    """See the module docstring. Feed it with `submit_image`, `submit_depth` (a camera frame
    callback) and `submit_poses` (a pipeline stage's frames callback)."""

    def __init__(self, settings: CaptureSyncSettings) -> None:
        super().__init__(daemon=True, name='CaptureSync')
        self.settings: CaptureSyncSettings = settings
        self._lock: Lock = Lock()
        self._wake: Condition = Condition(self._lock)
        self._running: bool = False
        self._images: dict[int, deque[GpuImage]] = defaultdict(lambda: deque(maxlen=KEPT))
        self._depths: dict[int, deque[CameraFrame]] = defaultdict(lambda: deque(maxlen=KEPT))
        self._poses: dict[int, deque[tuple[float, FrameDict]]] = defaultdict(lambda: deque(maxlen=KEPT))
        self._last: dict[int, float] = {}               # the capture time of each camera's last capture
        self._players: dict[int, bool] = {}             # poses came for the camera since a capture last went without
        self._waiting: dict[int, _Waiting] = {}
        self._misses: int = 0
        self._callbacks: list[CaptureCallback] = []     # run in registration order

    def add_capture_callback(self, callback: CaptureCallback) -> None:
        if callback not in self._callbacks:
            self._callbacks.append(callback)

    # ── Lifecycle ──────────────────────────────────────────────────────

    def start(self) -> None:
        self._running = True
        super().start()

    def stop(self) -> None:
        with self._lock:
            self._running = False
            self._wake.notify()
        if self.is_alive():
            self.join(timeout=2.0)

    def run(self) -> None:
        """Send out a waiting capture whose deadline has passed, with the poses from before it."""
        while True:
            with self._lock:
                capture: Capture | None = None
                while self._running and capture is None:
                    capture = self._overdue()
                if not self._running:
                    return
            self._emit(capture)

    # ── Inputs ─────────────────────────────────────────────────────────

    def submit_image(self, image: GpuImage) -> None:
        """An uploaded color image (`Uploader`'s callback, the camera's thread)."""
        with self._lock:
            self._images[image.cam_id].append(image)
            capture: Capture | None = self._match(image.cam_id)
        self._emit(capture)

    def submit_depth(self, frame: CameraFrame) -> None:
        """A camera frame callback: the DEPTH frames, the others pass."""
        if frame.kind != FrameType.DEPTH:
            return
        with self._lock:
            self._depths[frame.cam_id].append(frame)
            capture: Capture | None = self._match(frame.cam_id)
        self._emit(capture)

    def submit_poses(self, poses: FrameDict) -> None:
        """A pipeline stage's poses, each camera's stamped with its image's time."""
        by_cam: dict[int, FrameDict] = defaultdict(dict)
        for track_id, pose in poses.items():
            by_cam[pose.cam_id][track_id] = pose
        captures: list[Capture | None] = []
        with self._lock:
            for cam_id, cam_poses in by_cam.items():
                self._poses[cam_id].append((max(pose.timestamp for pose in cam_poses.values()), cam_poses))
                self._players[cam_id] = True
                captures.append(self._match(cam_id))
        for capture in captures:
            self._emit(capture)

    # ── Matching, under the lock ───────────────────────────────────────

    def _match(self, cam_id: int) -> Capture | None:
        """The newest moment of this camera with its image and its depth, newer than its last capture:
        out with its poses, or waiting for them while a player is in view."""
        last: float = self._last.get(cam_id, float('-inf'))
        for depth in reversed(self._depths[cam_id]):
            if depth.timestamp <= last + CAPTURE_TOLERANCE:
                break
            image: GpuImage | None = self._image_at(cam_id, depth.timestamp)
            if image is None:
                continue
            poses: FrameDict | None = self._poses_at(cam_id, depth.timestamp)
            if poses is not None:
                return self._capture(image, depth, poses, True)
            pose_wait: float = self.settings.pose_wait / 1000.0
            if self._players.get(cam_id, False) and pose_wait > 0.0:
                waiting: _Waiting | None = self._waiting.get(cam_id)
                if waiting is None or waiting.depth is not depth:
                    self._waiting[cam_id] = _Waiting(image, depth, time.perf_counter() + pose_wait)
                    self._wake.notify()
                return None
            return self._capture(image, depth, self._poses_before(cam_id, depth.timestamp), False)
        return None

    def _overdue(self) -> Capture | None:
        """The waiting capture whose deadline has passed, after waiting for the earliest one."""
        if not self._waiting:
            self._wake.wait()
            return None
        cam_id, waiting = min(self._waiting.items(), key=lambda item: item[1].deadline)
        remaining: float = waiting.deadline - time.perf_counter()
        if remaining > 0.0:
            self._wake.wait(remaining)
            return None
        self._players[cam_id] = False       # no poses in time: until poses come again, nobody is waited for
        return self._capture(waiting.image, waiting.depth, self._poses_before(cam_id, waiting.depth.timestamp), False)

    def _capture(self, image: GpuImage, depth: CameraFrame, poses: FrameDict, synced: bool) -> Capture:
        cam_id: int = depth.cam_id
        timestamp: float = depth.timestamp
        self._last[cam_id] = timestamp
        self._waiting.pop(cam_id, None)
        depths: deque[CameraFrame] = self._depths[cam_id]
        while depths and depths[0].timestamp <= timestamp + CAPTURE_TOLERANCE:
            if depths[0].timestamp < timestamp - CAPTURE_TOLERANCE:
                self._misses += 1               # an older depth frame never met its image
            depths.popleft()
        images: deque[GpuImage] = self._images[cam_id]
        while images and images[0].timestamp <= timestamp + CAPTURE_TOLERANCE:
            images.popleft()
        return Capture(cam_id=cam_id, timestamp=timestamp, image=image, depth=depth, poses=poses, poses_synced=synced)

    def _image_at(self, cam_id: int, timestamp: float) -> GpuImage | None:
        for image in self._images[cam_id]:
            if abs(image.timestamp - timestamp) <= CAPTURE_TOLERANCE:
                return image
        return None

    def _poses_at(self, cam_id: int, timestamp: float) -> FrameDict | None:
        for pose_time, poses in self._poses[cam_id]:
            if abs(pose_time - timestamp) <= CAPTURE_TOLERANCE:
                return poses
        return None

    def _poses_before(self, cam_id: int, timestamp: float) -> FrameDict:
        """The newest poses captured before `timestamp`; none when there are none."""
        for pose_time, poses in reversed(self._poses[cam_id]):
            if pose_time < timestamp:
                return poses
        return {}

    # ── Output ─────────────────────────────────────────────────────────

    def _emit(self, capture: Capture | None) -> None:
        """A complete capture to the callbacks, outside the lock."""
        if capture is None:
            return
        if self.settings.misses != self._misses:
            self.settings.misses = self._misses
        for callback in self._callbacks:
            try:
                callback(capture)
            except Exception:
                logger.exception('Error in capture callback')
