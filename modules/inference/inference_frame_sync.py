"""InferenceFrameSync: one camera moment's image and the results of it that its app runs, brought together as an
`InferenceFrame`.

The sync is made with the result types its app runs (`Depth`, `Poses`, `Masks`). They arrive on their own threads at
their own times: the image and the depth as they are uploaded (the uploader's thread, the depth about 13 ms after its
image), the poses out of a pipeline stage (the model's thread) and the masks out of the segmentation (its thread), both
measured to arrive about with the depth. Each is kept a few frames per camera and matched by capture time within
`CAPTURE_TOLERANCE`. A frame is complete with its image and, where the app has depth, its depth. While a model's
results keep up, the newest as new as the last frame, the frame waits for its own up to that type's wait (`pose_wait`,
`mask_wait`); otherwise (nobody in view, the model behind, or no poses asked for the image: `expect`) it takes the
newest from before it at once, so a missing result never holds it longer. Every complete frame goes out, in the order
of capture: one still waiting goes out with what it has as soon as a newer one is complete, so a late part never costs
a frame.

The submits only queue what they take, on the threads they come on, at most `KEPT` of a stream per camera; beyond
that the oldest is dropped, counted as `dropped`. The sync's own thread matches the parts in the order they came,
waits for the next deadline in the same wait, and hands the frames on. A part before `start` is an error; one after
`stop` is dropped. The parts are kept and found by capture time in `modules.camera.CaptureStream`s; an image or a
depth no frame took is counted and warned of (`MissedParts`), the depths without an image as `misses`.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, replace
from enum import IntEnum, auto
from threading import Event, Lock, Thread
from typing import Any, Callable

from modules.camera import CAPTURE_TOLERANCE, CaptureStream, MissedParts
from modules.settings import BaseSettings, Field
from modules.utils import Broadcast, PerformanceTimer, Waker

from .inference_frame import Depth, InferenceFrame, InferenceFrameCallback, InferenceResult, Masks, Poses
from .preprocessing import GpuImage

logger = logging.getLogger(__name__)

KEPT: int = 8       # frames of each stream kept per camera, and queued per camera: depth trails its image by about one


class InferenceFrameSyncSettings(BaseSettings):
    """Configuration for InferenceFrameSync."""
    pose_wait: Field[float] = Field(10.0, min=0.0, max=50.0, step=0.5, description="Longest a frame waits for its own poses (ms)")
    mask_wait: Field[float] = Field(8.0, min=0.0, max=50.0, step=0.5, description="Longest a frame waits for its own masks (ms)")
    misses:    Field[int]   = Field(0, access=Field.READ, description="Depth frames that found no image")
    dropped:   Field[int]   = Field(0, access=Field.READ, description="Parts dropped while the sync fell behind")


# The longest a frame waits for its own result of a type (ms), per type a model gives; the depth is not waited for
# but needed: a frame is complete with it
WAITS: dict[type[InferenceResult], Callable[[InferenceFrameSyncSettings], float]] = {
    Poses: lambda settings: settings.pose_wait,
    Masks: lambda settings: settings.mask_wait,
}


class _Part(IntEnum):
    """What waits in the sync's inbox."""
    IMAGE = 0
    RESULT = auto()     # an InferenceResult of one of the sync's types
    ASKED = auto()      # (timestamp, whether poses were asked for the camera's image)
    SETTLE = auto()     # an Event, set once every part queued before it is matched


@dataclass
class _Waiting:
    """A frame whose image (and depth) are in, waiting for its models' results until `deadline` (perf_counter)."""
    image: GpuImage
    depth: Depth | None
    deadline: float


class InferenceFrameSync:
    """See the module docstring. Feed it with `submit_image` (the uploader's image callback) and `submit` (each result:
    `Depth.of` the uploader's depth, `Poses.by_camera` a pipeline stage's poses, `Masks.by_camera` the segmentation's
    masks)."""

    def __init__(self, settings: InferenceFrameSyncSettings, results: tuple[type[InferenceResult], ...]) -> None:
        self._thread: Thread = Thread(target=self._run, daemon=True, name='InferenceFrameSync')
        for kind in results:
            if kind is not Depth and kind not in WAITS:
                raise ValueError(f"InferenceFrameSync: no wait for {kind.__name__}")
        self.settings: InferenceFrameSyncSettings = settings
        self._has_depth: bool = Depth in results
        self._waited: tuple[type[InferenceResult], ...] = tuple(kind for kind in results if kind is not Depth)
        self._lock: Lock = Lock()                       # the inbox
        self._waker: Waker = Waker()
        self._running: bool = False
        self._inbox: deque[tuple[_Part, int, Any, float]] = deque()      # each with the moment it was queued
        self._wait: PerformanceTimer = PerformanceTimer(name="Wait inference frame sync")
        self._queued: dict[tuple[_Part, int, type | None], int] = defaultdict(int)     # what the inbox holds per stream and camera
        self._dropped: int = 0
        self._stopped: bool = False
        # The sync's own, touched by one thread at a time
        self._images: dict[int, CaptureStream[GpuImage]] = defaultdict(lambda: CaptureStream(KEPT))
        self._results: dict[type[InferenceResult], dict[int, CaptureStream[InferenceResult]]] = {
            kind: defaultdict(lambda: CaptureStream(KEPT)) for kind in results}
        self._asks: dict[int, CaptureStream[bool]] = defaultdict(lambda: CaptureStream(KEPT))
        self._last: dict[int, float] = {}               # the capture time of each camera's last frame
        self._waiting: dict[int, _Waiting] = {}
        self._missed: MissedParts = MissedParts('Inference frame sync')
        self._callbacks: Broadcast[InferenceFrame] = Broadcast(name="inference frame")

    def add_frame_callback(self, callback: InferenceFrameCallback) -> None:
        self._callbacks.add_callback(callback)

    # ── Lifecycle ──────────────────────────────────────────────────────

    def start(self) -> None:
        with self._lock:
            self._running = True
        self._thread.start()

    def stop(self) -> None:
        """The thread ends; what still waits in the inbox is dropped and logged."""
        with self._lock:
            self._running = False
            self._stopped = True
            dropped: list[tuple[_Part, int, Any, float]] = list(self._inbox)
            self._inbox.clear()
            self._queued.clear()
        self._waker.set()
        parts: int = 0
        for part, _, value, _ in dropped:
            if part == _Part.SETTLE:
                value.set()             # nobody waits forever on a sync that stopped
            else:
                parts += 1
        if parts:
            logger.info("InferenceFrameSync: %d parts dropped at stop", parts)
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        if not self._thread.is_alive():
            self._waker.close()

    def _run(self) -> None:
        """Match what came in, in the order it came, and send out a waiting frame whose deadline has passed."""
        while True:
            with self._lock:
                if not self._running:
                    return
                parts: list[tuple[_Part, int, Any, float]] = list(self._inbox)
                self._inbox.clear()
                self._queued.clear()
                deadline: float | None = self._deadline()
            if not parts and (deadline is None or deadline > time.perf_counter()):
                self._waker.wait(deadline)
                continue
            try:
                taken: float = time.perf_counter()
                frames: list[InferenceFrame] = []
                for part, cam_id, value, queued in parts:
                    if part != _Part.SETTLE:
                        self._wait.add_time((taken - queued) * 1000.0, report=False)
                    frames += self._take(part, cam_id, value)
                frames += self._overdue()
                self._emit(frames)
            except Exception:
                logger.exception("InferenceFrameSync: matching failed")
                for part, _, value, _ in parts:
                    if part == _Part.SETTLE:
                        value.set()

    def settle(self) -> None:
        """Return once every part submitted before is matched and its frame, if complete, handed on."""
        done: Event = Event()
        self._queue(_Part.SETTLE, -1, done)
        done.wait()

    # ── Inputs: each queues and returns ────────────────────────────────

    def submit_image(self, image: GpuImage) -> None:
        """An uploaded color image (the uploader's callback, the uploader's thread)."""
        self._queue(_Part.IMAGE, image.cam_id, image)

    def submit(self, *results: InferenceResult) -> None:
        """Results of the sync's types, each stamped with its image's capture time; a type it was not made with is an
        error."""
        for result in results:
            if type(result) not in self._results:
                raise ValueError(f"InferenceFrameSync: not made with {type(result).__name__}")
            self._queue(_Part.RESULT, result.cam_id, result)

    def expect(self, cam_id: int, timestamp: float, asked: bool) -> None:
        """Whether poses were asked for the camera's image captured at `timestamp` (the frame source's asked
        callback): a frame whose poses were not asked waits for no model's result, as the models run on the poses."""
        self._queue(_Part.ASKED, cam_id, (timestamp, asked))

    def _queue(self, part: _Part, cam_id: int, value: Any) -> None:
        """Into the inbox, the stream's oldest dropped beyond `KEPT`. Before `start` an error; after `stop` dropped (a
        settle returns at once)."""
        with self._lock:
            if not self._running:
                if self._stopped:
                    if part == _Part.SETTLE:
                        value.set()
                    return
                raise RuntimeError("InferenceFrameSync: a part before start")
            kind: type | None = type(value) if part == _Part.RESULT else None
            key: tuple[_Part, int, type | None] = (part, cam_id, kind)
            if part != _Part.SETTLE and self._queued[key] >= KEPT:
                for index, (queued, queued_cam, queued_value, _) in enumerate(self._inbox):
                    if queued == part and queued_cam == cam_id and (kind is None or type(queued_value) is kind):
                        del self._inbox[index]
                        self._dropped += 1
                        break
            else:
                self._queued[key] += 1
            self._inbox.append((part, cam_id, value, time.perf_counter()))
        self._waker.set()

    # ── Matching, on the sync's thread ─────────────────────────────────

    def _take(self, part: _Part, cam_id: int, value: Any) -> list[InferenceFrame]:
        if part == _Part.SETTLE:
            value.set()
            return []
        if part == _Part.IMAGE:
            self._images[cam_id].add(value.timestamp, value)
        elif part == _Part.RESULT:
            self._results[type(value)][cam_id].add(value.timestamp, value)
        else:
            self._asks[cam_id].add(*value)
        return self._match(cam_id)

    def _complete(self, cam_id: int, last: float) -> list[tuple[GpuImage, Depth | None]]:
        """This camera's moments newer than `last` with all they need: the image, and its depth where the app has
        depth."""
        complete: list[tuple[GpuImage, Depth | None]] = []
        if self._has_depth:
            for depth_time, depth in self._results[Depth][cam_id]:
                if depth_time <= last + CAPTURE_TOLERANCE:
                    continue
                assert isinstance(depth, Depth)
                image: GpuImage | None = self._images[cam_id].at(depth_time)
                if image is not None:
                    complete.append((image, depth))
        else:
            for image_time, image in self._images[cam_id]:
                if image_time > last + CAPTURE_TOLERANCE:
                    complete.append((image, None))
        return complete

    def _match(self, cam_id: int) -> list[InferenceFrame]:
        """This camera's complete moments newer than its last frame, oldest first: each out with its results, or the
        newest from before, while it waits for those that keep up and are still on their way; one that waits goes
        out with what it has as soon as a newer moment is complete."""
        frames: list[InferenceFrame] = []
        last: float = self._last.get(cam_id, float('-inf'))
        complete: list[tuple[GpuImage, Depth | None]] = self._complete(cam_id, last)
        for index, (image, depth) in enumerate(complete):
            timestamp: float = depth.timestamp if depth is not None else image.timestamp
            wait: float = 0.0
            unasked: bool = self._asks[cam_id].at(timestamp) is False
            if index + 1 == len(complete) and not unasked:      # the newest may wait, for poses asked; an older one goes out now
                for kind in self._waited:
                    stream: CaptureStream[InferenceResult] = self._results[kind][cam_id]
                    if stream.at(timestamp) is None and stream.keeps_up(last):
                        wait = max(wait, WAITS[kind](self.settings) / 1000.0)
            if wait > 0.0:
                waiting: _Waiting | None = self._waiting.get(cam_id)
                if waiting is None or waiting.image is not image:
                    self._waiting[cam_id] = _Waiting(image, depth, time.perf_counter() + wait)
                break
            frames.append(self._frame(image, depth))
            last = timestamp
        return frames

    def _deadline(self) -> float | None:
        """The earliest waiting frame's deadline (perf_counter); None while none waits."""
        if not self._waiting:
            return None
        return min(waiting.deadline for waiting in self._waiting.values())

    def _overdue(self) -> list[InferenceFrame]:
        """The waiting frames whose deadline has passed."""
        now: float = time.perf_counter()
        return [self._frame(waiting.image, waiting.depth) for waiting in list(self._waiting.values()) if waiting.deadline <= now]

    def _frame(self, image: GpuImage, depth: Depth | None) -> InferenceFrame:
        """The frame of that moment, with each model's own result where it came in, else the newest from before it."""
        cam_id: int = image.cam_id
        timestamp: float = depth.timestamp if depth is not None else image.timestamp
        results: list[InferenceResult] = [depth] if depth is not None else []
        for kind in self._waited:
            stream: CaptureStream[InferenceResult] = self._results[kind][cam_id]
            own: InferenceResult | None = stream.at(timestamp)
            result: InferenceResult | None = own if own is not None else stream.before(timestamp)
            if result is not None:
                results.append(result if own is not None else replace(result, synced=False))
        self._last[cam_id] = timestamp
        self._waiting.pop(cam_id, None)
        # What came before this frame and was not taken never will be: a complete one would have gone out before
        if self._has_depth:
            self._missed.count(cam_id, 'image', self._results[Depth][cam_id].take_through(timestamp))     # a depth without its image
            self._missed.count(cam_id, 'depth', self._images[cam_id].take_through(timestamp))             # an image without its depth
        else:
            self._images[cam_id].take_through(timestamp)
        self._missed.started(cam_id)
        return InferenceFrame(image, results, timestamp)

    # ── Output ─────────────────────────────────────────────────────────

    def _emit(self, frames: list[InferenceFrame]) -> None:
        """Complete frames to the callbacks, in order."""
        if not frames:
            return
        misses: int = self._missed.total('image') + self._missed.at_start('image')
        if self.settings.misses != misses:
            self.settings.misses = misses
        with self._lock:
            dropped: int = self._dropped
        if self.settings.dropped != dropped:
            self.settings.dropped = dropped
        self._missed.warn()
        for frame in frames:
            self._callbacks(frame)
