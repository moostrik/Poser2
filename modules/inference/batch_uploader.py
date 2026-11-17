"""The batch uploader: a moment's frames of several cameras onto the GPU as one batch, for cameras that are
synced first.

`submit_set` takes the set the frames' `CameraSync` formed, each camera's color frame of one moment by `cam_id`, and
returns: sets wait whole, at most `WAITING`; beyond that the oldest set is dropped whole and counted (`dropped`), so
a frame never leaves its set. This one's own thread uploads a set with `ImageUpload`, as one stack when its frames
share a size (one copy, one round of kernels), and hands on each image to the image callbacks (`GpuImage`) and the
set to the snapshot callbacks (`ImageSnapshot`: the set's images, the previous set's as each camera's previous one,
their capture times, the event recorded after them). No tracklets and no depth: they do not go through it."""

import logging
import time
from collections import deque
from threading import Condition, Lock, Thread

import torch

from modules.camera import CameraFrame
from modules.utils import PerformanceTimer

from .image_upload import ImageUpload
from .source import GpuImage, GpuImageCallback, ImageDict, ImageSnapshot, ImageSnapshotCallback

logger = logging.getLogger(__name__)

WAITING: int = 2        # sets waiting for this thread; beyond that the oldest is dropped whole and counted


class BatchUploader(Thread):
    """See the module docstring."""

    def __init__(self) -> None:
        super().__init__(daemon=True, name='BatchUploader')
        self._image_upload: ImageUpload = ImageUpload()
        self._stream: torch.cuda.Stream = torch.cuda.Stream()
        self._previous: ImageDict = {}                  # each camera's image of the set before, on this thread
        self._wake: Condition = Condition()
        self._running: bool = False
        self._sets: deque[tuple[dict[int, CameraFrame], float]] = deque()      # each with the moment it was queued
        self._dropped: int = 0
        self._set_wait: PerformanceTimer = PerformanceTimer(name="Wait upload set")
        self._timer: PerformanceTimer = PerformanceTimer(name="GPU Set Upload", sample_count=200, report_interval=100,
                                                         color="green", omit_init=25)
        self._callback_lock: Lock = Lock()
        self._image_callbacks: list[GpuImageCallback] = []         # run in registration order, on this thread
        self._snapshot_callbacks: list[ImageSnapshotCallback] = [] # run in registration order, after the image callbacks

    @property
    def dropped(self) -> int:
        """Sets dropped whole because this thread had not taken the ones before them yet."""
        return self._dropped

    def add_gpu_image_callback(self, callback: GpuImageCallback) -> None:
        """Every uploaded image, with its capture time and the event that says it is written, on this thread."""
        with self._callback_lock:
            if callback not in self._image_callbacks:
                self._image_callbacks.append(callback)

    def add_snapshot_callback(self, callback: ImageSnapshotCallback) -> None:
        """Every uploaded set as an `ImageSnapshot`, on this thread, after the image callbacks."""
        with self._callback_lock:
            if callback not in self._snapshot_callbacks:
                self._snapshot_callbacks.append(callback)

    # ── Lifecycle ─────────────────────────────────────────────────────

    def start(self) -> None:
        with self._wake:
            self._running = True
        super().start()

    def stop(self) -> None:
        """The thread ends; the sets still waiting are dropped and logged."""
        with self._wake:
            self._running = False
            waiting: int = len(self._sets)
            self._sets.clear()
            self._wake.notify()
        if waiting:
            logger.info("BatchUploader: %d sets dropped at stop", waiting)
        if self.is_alive():
            self.join(timeout=2.0)

    def run(self) -> None:
        while True:
            with self._wake:
                while self._running and not self._sets:
                    self._wake.wait()
                if not self._running:
                    return
                frames, queued = self._sets.popleft()
            self._set_wait.add_time((time.perf_counter() - queued) * 1000.0, report=False)
            try:
                self._upload(frames)
            except Exception:
                logger.exception("BatchUploader: a set of cameras %s failed", sorted(frames))

    # ── Input: queues and returns ─────────────────────────────────────

    def submit_set(self, frames: dict[int, CameraFrame]) -> None:
        """A moment's color frames by `cam_id` (the frames' `CameraSync`'s set callback): queued whole, the oldest set
        dropped whole beyond `WAITING`."""
        with self._wake:
            if len(self._sets) >= WAITING:
                self._sets.popleft()
                self._dropped += 1
            self._sets.append((dict(frames), time.perf_counter()))
            self._wake.notify()

    # ── The upload, on this thread ────────────────────────────────────

    def _upload(self, frames: dict[int, CameraFrame]) -> None:
        start: float = time.perf_counter()
        # One stack per size and layout: the four cameras of a rig make one
        stacks: dict[tuple, list[CameraFrame]] = {}
        for frame in frames.values():
            stacks.setdefault((frame.image.shape, frame.image.dtype, frame.nv12), []).append(frame)
        images: ImageDict = {}
        ready: torch.cuda.Event | None = None
        for stack in stacks.values():
            uploaded, ready = self._image_upload.upload(stack, self._stream)
            for frame, image in zip(stack, uploaded):
                images[frame.cam_id] = image
        assert ready is not None        # recorded after the last stack, so after every image of the set
        previous: ImageDict = {cam_id: self._previous[cam_id] for cam_id in images if cam_id in self._previous}
        self._previous.update(images)
        self._timer.add_time((time.perf_counter() - start) * 1000.0, report=False)

        timestamps: dict[int, float] = {cam_id: frame.timestamp for cam_id, frame in frames.items()}
        snapshot: ImageSnapshot = (images, previous, timestamps, ready)
        with self._callback_lock:
            image_callbacks: list[GpuImageCallback] = list(self._image_callbacks)
            snapshot_callbacks: list[ImageSnapshotCallback] = list(self._snapshot_callbacks)
        for cam_id, image in images.items():
            gpu_image: GpuImage = GpuImage(cam_id, timestamps[cam_id], image, ready, frames[cam_id])
            for callback in image_callbacks:
                try:
                    callback(gpu_image)
                except Exception:
                    logger.exception("BatchUploader: error in an image callback")
        for snapshot_callback in snapshot_callbacks:
            try:
                snapshot_callback(snapshot)
            except Exception:
                logger.exception("BatchUploader: error in a snapshot callback")
