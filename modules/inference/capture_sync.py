"""CaptureSync: the color image, the depth frame, the poses and RVM's masks of one moment of capture, brought together.

They arrive on their own threads at their own times: the image and the depth frame as they are uploaded
(the uploader's thread, the depth about 13 ms after its image), the poses out of a pipeline stage
(the model's thread) and the masks out of RVM (its thread), both measured to arrive about with the
depth. Each is kept a few frames per camera and matched by capture time within `CAPTURE_TOLERANCE`.
A capture is complete with its image and its depth. While the poses keep up, the newest as new as
the last capture, it waits up to `pose_wait` for its own, and while the masks keep up, up to
`mask_wait` for its own; otherwise (nobody in view, no masks, either behind, or its poses not asked:
`expect_poses`) it takes the newest from before it at once, so a missing pose or mask never holds it
longer. Every complete capture goes
out, in the order of capture: one still waiting goes out with what it has as soon as a newer one is
complete, so a late part never costs a capture.

The submits only queue what they take, on the threads they come on, at most `KEPT` of a stream per camera; beyond
that the oldest is dropped, counted as `dropped`. The sync's own thread matches the parts in the order they came,
waits for the next deadline in the same wait, and hands the captures on. A part before `start` is an error; one
after `stop` is dropped. The parts are kept and found by capture time in `modules.camera.CaptureStream`s;
an image or a depth no capture took is counted and warned of (`MissedParts`), the depths without an image as
`misses`.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import IntEnum, auto
from threading import Condition, Event, Lock, Thread
from typing import TYPE_CHECKING, Any

from modules.camera import CAPTURE_TOLERANCE, CaptureStream, MissedParts
from modules.pose import FrameDict
from modules.settings import BaseSettings, Field
from modules.utils import PerformanceTimer

from .capture import Capture, CaptureCallback
from .preprocessing import GpuDepth, GpuImage

if TYPE_CHECKING:
    from .segmentation import ImageDict

logger = logging.getLogger(__name__)

KEPT: int = 8       # frames of each stream kept per camera, and queued per camera: depth trails its image by about one


class CaptureSyncSettings(BaseSettings):
    """Configuration for CaptureSync."""
    pose_wait: Field[float] = Field(10.0, min=0.0, max=50.0, step=0.5, description="Longest a capture waits for its own poses (ms)")
    mask_wait: Field[float] = Field(8.0, min=0.0, max=50.0, step=0.5, description="Longest a capture waits for its own masks (ms)")
    misses:    Field[int]   = Field(0, access=Field.READ, description="Depth frames that found no image")
    dropped:   Field[int]   = Field(0, access=Field.READ, description="Parts dropped while the sync fell behind")


class _Part(IntEnum):
    """What waits in the sync's inbox."""
    IMAGE = 0
    DEPTH = auto()
    POSES = auto()      # (timestamp, one camera's poses)
    ASKED = auto()      # (timestamp, whether poses were asked for the camera's image)
    MASKS = auto()      # (timestamp, one camera's masks)
    SETTLE = auto()     # an Event, set once every part queued before it is matched


@dataclass
class _Waiting:
    """A capture whose image and depth are in, waiting for its poses or masks until `deadline` (perf_counter)."""
    image: GpuImage
    depth: GpuDepth
    deadline: float


class CaptureSync(Thread):
    """See the module docstring. Feed it with `submit_image` and `submit_depth` (the uploader's
    callbacks), `submit_poses` (a pipeline stage's frames callback) and `submit_masks` (RVM's
    segmentation callback)."""

    def __init__(self, settings: CaptureSyncSettings) -> None:
        super().__init__(daemon=True, name='CaptureSync')
        self.settings: CaptureSyncSettings = settings
        self._lock: Lock = Lock()                       # the inbox
        self._wake: Condition = Condition(self._lock)
        self._running: bool = False
        self._inbox: deque[tuple[_Part, int, Any, float]] = deque()      # each with the moment it was queued
        self._wait: PerformanceTimer = PerformanceTimer(name="Wait capture sync")
        self._queued: dict[tuple[_Part, int], int] = defaultdict(int)     # what the inbox holds per stream and camera
        self._dropped: int = 0
        self._stopped: bool = False
        # The sync's own, touched by one thread at a time
        self._images: dict[int, CaptureStream[GpuImage]] = defaultdict(lambda: CaptureStream(KEPT))
        self._depths: dict[int, CaptureStream[GpuDepth]] = defaultdict(lambda: CaptureStream(KEPT))
        self._poses: dict[int, CaptureStream[FrameDict]] = defaultdict(lambda: CaptureStream(KEPT))
        self._asks: dict[int, CaptureStream[bool]] = defaultdict(lambda: CaptureStream(KEPT))
        self._masks: dict[int, CaptureStream[ImageDict]] = defaultdict(lambda: CaptureStream(KEPT))
        self._last: dict[int, float] = {}               # the capture time of each camera's last capture
        self._waiting: dict[int, _Waiting] = {}
        self._missed: MissedParts = MissedParts('Capture sync')
        self._callback_lock: Lock = Lock()
        self._callbacks: list[CaptureCallback] = []     # run in registration order

    def add_capture_callback(self, callback: CaptureCallback) -> None:
        with self._callback_lock:
            if callback not in self._callbacks:
                self._callbacks.append(callback)

    # ── Lifecycle ──────────────────────────────────────────────────────

    def start(self) -> None:
        with self._lock:
            self._running = True
        super().start()

    def stop(self) -> None:
        """The thread ends; what still waits in the inbox is dropped and logged."""
        with self._lock:
            self._running = False
            self._stopped = True
            dropped: list[tuple[_Part, int, Any, float]] = list(self._inbox)
            self._inbox.clear()
            self._queued.clear()
            self._wake.notify()
        parts: int = 0
        for part, _, value, _ in dropped:
            if part == _Part.SETTLE:
                value.set()             # nobody waits forever on a sync that stopped
            else:
                parts += 1
        if parts:
            logger.info("CaptureSync: %d parts dropped at stop", parts)
        if self.is_alive():
            self.join(timeout=2.0)

    def run(self) -> None:
        """Match what came in, in the order it came, and send out a waiting capture whose deadline has passed."""
        while True:
            with self._lock:
                while self._running and not self._inbox:
                    remaining: float | None = self._remaining()
                    if remaining is not None and remaining <= 0.0:
                        break
                    self._wake.wait(remaining)
                if not self._running:
                    return
                parts: list[tuple[_Part, int, Any, float]] = list(self._inbox)
                self._inbox.clear()
                self._queued.clear()
            try:
                taken: float = time.perf_counter()
                captures: list[Capture] = []
                for part, cam_id, value, queued in parts:
                    if part != _Part.SETTLE:
                        self._wait.add_time((taken - queued) * 1000.0, report=False)
                    captures += self._take(part, cam_id, value)
                captures += self._overdue()
                self._emit(captures)
            except Exception:
                logger.exception("CaptureSync: matching failed")
                for part, _, value, _ in parts:
                    if part == _Part.SETTLE:
                        value.set()

    def settle(self) -> None:
        """Return once every part submitted before is matched and its capture, if complete, handed on."""
        done: Event = Event()
        self._queue(_Part.SETTLE, -1, done)
        done.wait()

    # ── Inputs: each queues and returns ────────────────────────────────

    def submit_image(self, image: GpuImage) -> None:
        """An uploaded color image (the `SerialUploader`'s callback, the uploader's thread)."""
        self._queue(_Part.IMAGE, image.cam_id, image)

    def submit_depth(self, depth: GpuDepth) -> None:
        """An uploaded depth frame (the `SerialUploader`'s GPU depth callback, the uploader's thread)."""
        self._queue(_Part.DEPTH, depth.cam_id, depth)

    def submit_poses(self, poses: FrameDict) -> None:
        """A pipeline stage's poses, each camera's stamped with its image's time."""
        by_cam: dict[int, FrameDict] = defaultdict(dict)
        for track_id, pose in poses.items():
            by_cam[pose.cam_id][track_id] = pose
        for cam_id, cam_poses in by_cam.items():
            self._queue(_Part.POSES, cam_id, (max(pose.timestamp for pose in cam_poses.values()), cam_poses))

    def expect_poses(self, cam_id: int, timestamp: float, asked: bool) -> None:
        """Whether poses were asked for the camera's image captured at `timestamp` (the frame source's asked
        callback): a capture whose poses were not asked waits neither for poses nor for the masks cut for them."""
        self._queue(_Part.ASKED, cam_id, (timestamp, asked))

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
        for cam_id, cam_masks in by_cam.items():
            self._queue(_Part.MASKS, cam_id, (times[cam_id], cam_masks))

    def _queue(self, part: _Part, cam_id: int, value: Any) -> None:
        """Into the inbox, the stream's oldest dropped beyond `KEPT`. Before `start` an error; after `stop` dropped (a
        settle returns at once)."""
        with self._wake:
            if not self._running:
                if self._stopped:
                    if part == _Part.SETTLE:
                        value.set()
                    return
                raise RuntimeError("CaptureSync: a part before start")
            key: tuple[_Part, int] = (part, cam_id)
            if part != _Part.SETTLE and self._queued[key] >= KEPT:
                for index, (queued, queued_cam, _, _) in enumerate(self._inbox):
                    if queued == part and queued_cam == cam_id:
                        del self._inbox[index]
                        self._dropped += 1
                        break
            else:
                self._queued[key] += 1
            self._inbox.append((part, cam_id, value, time.perf_counter()))
            self._wake.notify()

    # ── Matching, on the sync's thread ─────────────────────────────────

    def _take(self, part: _Part, cam_id: int, value: Any) -> list[Capture]:
        if part == _Part.SETTLE:
            value.set()
            return []
        if part == _Part.IMAGE:
            self._images[cam_id].add(value.timestamp, value)
        elif part == _Part.DEPTH:
            self._depths[cam_id].add(value.timestamp, value)
        elif part == _Part.POSES:
            self._poses[cam_id].add(*value)
        elif part == _Part.ASKED:
            self._asks[cam_id].add(*value)
        else:
            self._masks[cam_id].add(*value)
        return self._match(cam_id)

    def _match(self, cam_id: int) -> list[Capture]:
        """This camera's complete moments newer than its last capture, oldest first: each out with its poses and
        masks, or the newest, while it waits for those that keep up and are still on their way; one that waits
        goes out with what it has as soon as a newer moment is complete."""
        captures: list[Capture] = []
        complete: list[tuple[GpuImage, GpuDepth]] = []
        last: float = self._last.get(cam_id, float('-inf'))
        for depth_time, depth in self._depths[cam_id]:
            if depth_time <= last + CAPTURE_TOLERANCE:
                continue
            image: GpuImage | None = self._images[cam_id].at(depth_time)
            if image is not None:
                complete.append((image, depth))
        for index, (image, depth) in enumerate(complete):
            wait: float = 0.0
            unasked: bool = self._asks[cam_id].at(depth.timestamp) is False
            if index + 1 == len(complete) and not unasked:      # the newest may wait, for poses asked; an older one goes out now
                poses, masks = self._poses[cam_id], self._masks[cam_id]
                if poses.at(depth.timestamp) is None and poses.keeps_up(last):
                    wait = max(wait, self.settings.pose_wait / 1000.0)
                if masks.at(depth.timestamp) is None and masks.keeps_up(last):
                    wait = max(wait, self.settings.mask_wait / 1000.0)
            if wait > 0.0:
                waiting: _Waiting | None = self._waiting.get(cam_id)
                if waiting is None or waiting.depth is not depth:
                    self._waiting[cam_id] = _Waiting(image, depth, time.perf_counter() + wait)
                break
            captures.append(self._capture(image, depth))
            last = depth.timestamp
        return captures

    def _remaining(self) -> float | None:
        """The seconds until the earliest waiting capture's deadline; None while none waits."""
        if not self._waiting:
            return None
        return min(waiting.deadline for waiting in list(self._waiting.values())) - time.perf_counter()

    def _overdue(self) -> list[Capture]:
        """The waiting captures whose deadline has passed."""
        now: float = time.perf_counter()
        return [self._capture(waiting.image, waiting.depth) for waiting in list(self._waiting.values()) if waiting.deadline <= now]

    def _capture(self, image: GpuImage, depth: GpuDepth) -> Capture:
        """The capture of `depth`'s moment, with its own poses and masks where they came in, else the newest from before it."""
        cam_id: int = depth.cam_id
        timestamp: float = depth.timestamp
        poses: FrameDict | None = self._poses[cam_id].at(timestamp)
        masks: ImageDict | None = self._masks[cam_id].at(timestamp)
        self._last[cam_id] = timestamp
        self._waiting.pop(cam_id, None)
        # What came before this capture and was not taken never will be: a complete one would have gone out before
        self._missed.count(cam_id, 'image', self._depths[cam_id].take_through(timestamp))     # a depth without its image
        self._missed.count(cam_id, 'depth', self._images[cam_id].take_through(timestamp))     # an image without its depth
        self._missed.started(cam_id)
        return Capture(cam_id=cam_id, timestamp=timestamp, image=image, depth=depth,
                       poses=poses if poses is not None else self._poses[cam_id].before(timestamp) or {}, poses_synced=poses is not None,
                       masks=masks if masks is not None else self._masks[cam_id].before(timestamp) or {}, masks_synced=masks is not None)

    # ── Output ─────────────────────────────────────────────────────────

    def _emit(self, captures: list[Capture]) -> None:
        """Complete captures to the callbacks, in order."""
        if not captures:
            return
        misses: int = self._missed.total('image') + self._missed.at_start('image')
        if self.settings.misses != misses:
            self.settings.misses = misses
        with self._lock:
            dropped: int = self._dropped
        if self.settings.dropped != dropped:
            self.settings.dropped = dropped
        self._missed.warn()
        with self._callback_lock:
            callbacks: list[CaptureCallback] = list(self._callbacks)
        for capture in captures:
            for callback in callbacks:
                try:
                    callback(capture)
                except Exception:
                    logger.exception('Error in capture callback')
