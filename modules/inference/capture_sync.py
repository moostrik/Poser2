"""CaptureSync: the color image, the depth frame, the poses and RVM's masks of one moment of capture, brought together.

They arrive on their own threads at their own times: the image as it is uploaded (the camera's
video thread), the depth frame about 13 ms later (its own thread), the poses out of a pipeline stage
(the model's thread) and the masks out of RVM (its thread), both measured to arrive about with the
depth. Each is kept a few frames per camera and matched by capture time within `CAPTURE_TOLERANCE`.
A capture is complete with its image and its depth. While the poses keep up, the newest as new as
the last capture, it waits up to `pose_wait` for its own, and while the masks keep up, up to
`mask_wait` for its own; otherwise (nobody in view, no masks, or either behind) it takes the newest
from before it at once, so a missing pose or mask never holds it longer. Every complete capture goes
out, in the order of capture: one still waiting goes out with what it has as soon as a newer one is
complete, so a late part never costs a capture. The wait runs on this sync's own thread: no submit
blocks, and a capture goes out on whichever thread completes it, under one lock, so they leave in order.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from threading import Condition, Lock, Thread
from typing import TYPE_CHECKING

from modules.camera import CAPTURE_TOLERANCE, CameraFrame, FrameType
from modules.pose import FrameDict
from modules.settings import BaseSettings, Field

from .capture import Capture, CaptureCallback
from .source import GpuImage

if TYPE_CHECKING:
    from .segmentation import ImageDict

logger = logging.getLogger(__name__)

KEPT: int = 8       # frames of each stream kept per camera: depth trails its image by about one


class CaptureSyncSettings(BaseSettings):
    """Configuration for CaptureSync."""
    pose_wait: Field[float] = Field(10.0, min=0.0, max=50.0, step=0.5, description="Longest a capture waits for its own poses (ms)")
    mask_wait: Field[float] = Field(8.0, min=0.0, max=50.0, step=0.5, description="Longest a capture waits for its own masks (ms)")
    misses:    Field[int]   = Field(0, access=Field.READ, description="Depth frames that found no image")


@dataclass
class _Waiting:
    """A capture whose image and depth are in, waiting for its poses or masks until `deadline` (perf_counter)."""
    image: GpuImage
    depth: CameraFrame
    deadline: float


class CaptureSync(Thread):
    """See the module docstring. Feed it with `submit_image`, `submit_depth` (a camera frame
    callback), `submit_poses` (a pipeline stage's frames callback) and `submit_masks` (RVM's
    segmentation callback)."""

    def __init__(self, settings: CaptureSyncSettings) -> None:
        super().__init__(daemon=True, name='CaptureSync')
        self.settings: CaptureSyncSettings = settings
        self._lock: Lock = Lock()
        self._wake: Condition = Condition(self._lock)
        self._running: bool = False
        self._images: dict[int, deque[GpuImage]] = defaultdict(lambda: deque(maxlen=KEPT))
        self._depths: dict[int, deque[CameraFrame]] = defaultdict(lambda: deque(maxlen=KEPT))
        self._poses: dict[int, deque[tuple[float, FrameDict]]] = defaultdict(lambda: deque(maxlen=KEPT))
        self._masks: dict[int, deque[tuple[float, ImageDict]]] = defaultdict(lambda: deque(maxlen=KEPT))
        self._last: dict[int, float] = {}               # the capture time of each camera's last capture
        self._waiting: dict[int, _Waiting] = {}
        self._misses: int = 0
        self._emit_lock: Lock = Lock()                  # matching and sending as one step, so captures leave in order
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
        """Send out a waiting capture whose deadline has passed, with what came in for it or before it."""
        while True:
            with self._lock:
                while self._running and not self._due():
                    pass
                if not self._running:
                    return
            with self._emit_lock:
                with self._lock:
                    captures: list[Capture] = self._overdue()
                self._emit(captures)

    # ── Inputs ─────────────────────────────────────────────────────────

    def submit_image(self, image: GpuImage) -> None:
        """An uploaded color image (`Uploader`'s callback, the uploader's thread)."""
        with self._emit_lock:
            with self._lock:
                self._images[image.cam_id].append(image)
                captures: list[Capture] = self._match(image.cam_id)
            self._emit(captures)

    def submit_depth(self, frame: CameraFrame) -> None:
        """A depth frame (`Uploader`'s depth callback); other kinds pass."""
        if frame.kind != FrameType.DEPTH:
            return
        with self._emit_lock:
            with self._lock:
                self._depths[frame.cam_id].append(frame)
                captures: list[Capture] = self._match(frame.cam_id)
            self._emit(captures)

    def submit_poses(self, poses: FrameDict) -> None:
        """A pipeline stage's poses, each camera's stamped with its image's time."""
        by_cam: dict[int, FrameDict] = defaultdict(dict)
        for track_id, pose in poses.items():
            by_cam[pose.cam_id][track_id] = pose
        with self._emit_lock:
            captures: list[Capture] = []
            with self._lock:
                for cam_id, cam_poses in by_cam.items():
                    self._poses[cam_id].append((max(pose.timestamp for pose in cam_poses.values()), cam_poses))
                    captures += self._match(cam_id)
            self._emit(captures)

    def submit_masks(self, poses: FrameDict, masks: ImageDict) -> None:
        """RVM's masks per track_id, with the poses they were cut for, each stamped with its image's time.
        A result without masks passes: the last ones stay, as on the board."""
        by_cam: dict[int, ImageDict] = defaultdict(dict)
        times: dict[int, float] = {}
        for track_id, mask in masks.items():
            pose = poses.get(track_id)
            if pose is None:
                continue
            by_cam[pose.cam_id][track_id] = mask
            times[pose.cam_id] = max(times.get(pose.cam_id, pose.timestamp), pose.timestamp)
        with self._emit_lock:
            captures: list[Capture] = []
            with self._lock:
                for cam_id, cam_masks in by_cam.items():
                    self._masks[cam_id].append((times[cam_id], cam_masks))
                    captures += self._match(cam_id)
            self._emit(captures)

    # ── Matching, under the lock ───────────────────────────────────────

    def _match(self, cam_id: int) -> list[Capture]:
        """This camera's complete moments newer than its last capture, oldest first: each out with its poses and
        masks, or the newest, while it waits for those that keep up and are still on their way; one that waits
        goes out with what it has as soon as a newer moment is complete."""
        captures: list[Capture] = []
        complete: list[tuple[GpuImage, CameraFrame]] = []
        last: float = self._last.get(cam_id, float('-inf'))
        for depth in self._depths[cam_id]:
            if depth.timestamp <= last + CAPTURE_TOLERANCE:
                continue
            image: GpuImage | None = self._image_at(cam_id, depth.timestamp)
            if image is not None:
                complete.append((image, depth))
        for index, (image, depth) in enumerate(complete):
            wait: float = 0.0
            if index + 1 == len(complete):          # the newest: it may wait; an older one goes out now
                if _at(self._poses[cam_id], depth.timestamp) is None and _keep_up(self._poses[cam_id], last):
                    wait = max(wait, self.settings.pose_wait / 1000.0)
                if _at(self._masks[cam_id], depth.timestamp) is None and _keep_up(self._masks[cam_id], last):
                    wait = max(wait, self.settings.mask_wait / 1000.0)
            if wait > 0.0:
                waiting: _Waiting | None = self._waiting.get(cam_id)
                if waiting is None or waiting.depth is not depth:
                    self._waiting[cam_id] = _Waiting(image, depth, time.perf_counter() + wait)
                    self._wake.notify()
                break
            captures.append(self._capture(image, depth))
            last = depth.timestamp
        return captures

    def _due(self) -> bool:
        """Whether a waiting capture's deadline has passed, after waiting for the earliest one."""
        if not self._waiting:
            self._wake.wait()
            return False
        waiting: _Waiting = min(self._waiting.values(), key=lambda item: item.deadline)
        remaining: float = waiting.deadline - time.perf_counter()
        if remaining > 0.0:
            self._wake.wait(remaining)
            return False
        return True

    def _overdue(self) -> list[Capture]:
        """The waiting captures whose deadline has passed, still waiting when this runs."""
        now: float = time.perf_counter()
        return [self._capture(waiting.image, waiting.depth) for waiting in list(self._waiting.values()) if waiting.deadline <= now]

    def _capture(self, image: GpuImage, depth: CameraFrame) -> Capture:
        """The capture of `depth`'s moment, with its own poses and masks where they came in, else the newest from before it."""
        cam_id: int = depth.cam_id
        timestamp: float = depth.timestamp
        poses: FrameDict | None = _at(self._poses[cam_id], timestamp)
        masks: ImageDict | None = _at(self._masks[cam_id], timestamp)
        self._last[cam_id] = timestamp
        self._waiting.pop(cam_id, None)
        depths: deque[CameraFrame] = self._depths[cam_id]
        while depths and depths[0].timestamp <= timestamp + CAPTURE_TOLERANCE:
            older: CameraFrame = depths.popleft()
            if older.timestamp < timestamp - CAPTURE_TOLERANCE:
                self._misses += 1               # it never met its image: a complete one would have gone out before
        images: deque[GpuImage] = self._images[cam_id]
        while images and images[0].timestamp <= timestamp + CAPTURE_TOLERANCE:
            images.popleft()
        return Capture(cam_id=cam_id, timestamp=timestamp, image=image, depth=depth,
                       poses=poses if poses is not None else _before(self._poses[cam_id], timestamp), poses_synced=poses is not None,
                       masks=masks if masks is not None else _before(self._masks[cam_id], timestamp), masks_synced=masks is not None)

    def _image_at(self, cam_id: int, timestamp: float) -> GpuImage | None:
        for image in self._images[cam_id]:
            if abs(image.timestamp - timestamp) <= CAPTURE_TOLERANCE:
                return image
        return None

    # ── Output ─────────────────────────────────────────────────────────

    def _emit(self, captures: list[Capture]) -> None:
        """Complete captures to the callbacks, in order, under the emit lock and outside the matching lock."""
        if not captures:
            return
        if self.settings.misses != self._misses:
            self.settings.misses = self._misses
        for capture in captures:
            for callback in self._callbacks:
                try:
                    callback(capture)
                except Exception:
                    logger.exception('Error in capture callback')


# ── A stream of (capture time, value), kept per camera ─────────────────

def _at[T](stream: deque[tuple[float, T]], timestamp: float) -> T | None:
    """The value captured at `timestamp`; None when it has not come in."""
    for value_time, value in stream:
        if abs(value_time - timestamp) <= CAPTURE_TOLERANCE:
            return value
    return None


def _before[T: dict](stream: deque[tuple[float, T]], timestamp: float) -> T | dict:
    """The newest value captured before `timestamp`; empty when there is none."""
    for value_time, value in reversed(stream):
        if value_time < timestamp:
            return value
    return {}


def _keep_up[T](stream: deque[tuple[float, T]], last: float) -> bool:
    """The stream's newest is as new as the camera's last capture: the next one is on its way.
    False while it is empty (nobody in view, no masks) and while it runs a frame or more behind."""
    return bool(stream) and stream[-1][0] >= last - CAPTURE_TOLERANCE
