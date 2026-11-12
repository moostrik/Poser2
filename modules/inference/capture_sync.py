"""CaptureSync: the color image, the depth frame, the poses and RVM's masks of one moment of capture, brought together.

They arrive on their own threads at their own times: the image and the depth frame as they are uploaded
(the uploader's thread, the depth about 13 ms after its image), the poses out of a pipeline stage
(the model's thread) and the masks out of RVM (its thread), both measured to arrive about with the
depth. Each is kept a few frames per camera and matched by capture time within `CAPTURE_TOLERANCE`.
A capture is complete with its image and its depth. While the poses keep up, the newest as new as
the last capture, it waits up to `pose_wait` for its own, and while the masks keep up, up to
`mask_wait` for its own; otherwise (nobody in view, no masks, or either behind) it takes the newest
from before it at once, so a missing pose or mask never holds it longer. Every complete capture goes
out, in the order of capture: one still waiting goes out with what it has as soon as a newer one is
complete, so a late part never costs a capture. The wait runs on this sync's own thread: no submit
blocks, and a capture goes out on whichever thread completes it, under one lock, so they leave in order.
The parts are kept and found by capture time in `modules.camera.CaptureStream`s; an image or a depth no capture
took is counted and warned of (`MissedParts`), the depths without an image as `misses`.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from threading import Condition, Lock, Thread
from typing import TYPE_CHECKING

from modules.camera import CAPTURE_TOLERANCE, CaptureStream, MissedParts
from modules.pose import FrameDict
from modules.settings import BaseSettings, Field

from .capture import Capture, CaptureCallback
from .source import GpuDepth, GpuImage

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
    depth: GpuDepth
    deadline: float


class CaptureSync(Thread):
    """See the module docstring. Feed it with `submit_image` and `submit_depth` (the uploader's
    callbacks), `submit_poses` (a pipeline stage's frames callback) and `submit_masks` (RVM's
    segmentation callback)."""

    def __init__(self, settings: CaptureSyncSettings) -> None:
        super().__init__(daemon=True, name='CaptureSync')
        self.settings: CaptureSyncSettings = settings
        self._lock: Lock = Lock()
        self._wake: Condition = Condition(self._lock)
        self._running: bool = False
        self._images: dict[int, CaptureStream[GpuImage]] = defaultdict(lambda: CaptureStream(KEPT))
        self._depths: dict[int, CaptureStream[GpuDepth]] = defaultdict(lambda: CaptureStream(KEPT))
        self._poses: dict[int, CaptureStream[FrameDict]] = defaultdict(lambda: CaptureStream(KEPT))
        self._masks: dict[int, CaptureStream[ImageDict]] = defaultdict(lambda: CaptureStream(KEPT))
        self._last: dict[int, float] = {}               # the capture time of each camera's last capture
        self._waiting: dict[int, _Waiting] = {}
        self._missed: MissedParts = MissedParts('Capture sync')
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
                self._images[image.cam_id].add(image.timestamp, image)
                captures: list[Capture] = self._match(image.cam_id)
            self._emit(captures)

    def submit_depth(self, depth: GpuDepth) -> None:
        """An uploaded depth frame (`Uploader`'s GPU depth callback, the uploader's thread)."""
        with self._emit_lock:
            with self._lock:
                self._depths[depth.cam_id].add(depth.timestamp, depth)
                captures: list[Capture] = self._match(depth.cam_id)
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
                    self._poses[cam_id].add(max(pose.timestamp for pose in cam_poses.values()), cam_poses)
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
                    self._masks[cam_id].add(times[cam_id], cam_masks)
                    captures += self._match(cam_id)
            self._emit(captures)

    # ── Matching, under the lock ───────────────────────────────────────

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
            if index + 1 == len(complete):          # the newest: it may wait; an older one goes out now
                poses, masks = self._poses[cam_id], self._masks[cam_id]
                if poses.at(depth.timestamp) is None and poses.keeps_up(last):
                    wait = max(wait, self.settings.pose_wait / 1000.0)
                if masks.at(depth.timestamp) is None and masks.keeps_up(last):
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
        """Complete captures to the callbacks, in order, under the emit lock and outside the matching lock."""
        if not captures:
            return
        misses: int = self._missed.total('image') + self._missed.at_start('image')
        if self.settings.misses != misses:
            self.settings.misses = misses
        self._missed.warn()         # outside the matching lock
        for capture in captures:
            for callback in self._callbacks:
                try:
                    callback(capture)
                except Exception:
                    logger.exception('Error in capture callback')
