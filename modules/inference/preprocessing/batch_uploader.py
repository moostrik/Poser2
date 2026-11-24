"""The batch uploader: a moment's frames of several cameras onto the GPU as one batch, for cameras that are
synced first.

`submit_set` takes the set the frames' `CameraSync` formed, each camera's color frame of one moment by `cam_id`, and
returns: sets wait whole, at most `WAITING`; beyond that the oldest set is dropped whole and counted (`dropped`), so
a frame never leaves its set. This one's own thread uploads a set with `ImageUpload`, as one stack when its frames
share a size (one copy, one round of kernels), and hands on each image to the image callbacks (`GpuImage`) and the
set to the snapshot callbacks (`ImageSnapshot`: the set's images, their capture times, the event recorded after
them), at once, for the readers in the package, which wait on the event on their own streams; then the set's
capture times to the capture callbacks; then, only when a finished callback is registered, it waits for its stream
and hands each image on written, for the readers outside the package. No tracklets and no depth: they do not go through it."""

import logging
import time
from collections import deque
from threading import Condition, Thread

import torch

from modules.camera import CameraFrame
from modules.utils import Broadcast, PerformanceTimer

from .image_upload import ImageUpload
from .source import CaptureCallback, GpuImage, GpuImageCallback, ImageDict, ImageSnapshot, ImageSnapshotCallback

logger = logging.getLogger(__name__)

WAITING: int = 2        # sets waiting for this thread; beyond that the oldest is dropped whole and counted


class BatchUploader:
    """See the module docstring."""

    def __init__(self) -> None:
        self._thread: Thread = Thread(target=self._run, daemon=True, name='BatchUploader')
        self._image_upload: ImageUpload = ImageUpload()
        self._stream: torch.cuda.Stream = torch.cuda.Stream()
        self._wake: Condition = Condition()
        self._running: bool = False
        self._sets: deque[tuple[dict[int, CameraFrame], float]] = deque()      # each with the moment it was queued
        self._dropped: int = 0
        self._set_wait: PerformanceTimer = PerformanceTimer(name="Wait upload set")
        self._timer: PerformanceTimer = PerformanceTimer(name="GPU Set Upload", sample_count=200, report_interval=100,
                                                         color="green", omit_init=25)
        self._image_callbacks: Broadcast[[GpuImage]] = Broadcast(name="BatchUploader image")
        self._snapshot_callbacks: Broadcast[[ImageSnapshot]] = Broadcast(name="BatchUploader snapshot")
        self._capture_callbacks: Broadcast[[dict[int, float]]] = Broadcast(name="BatchUploader capture")
        self._finished_image_callbacks: Broadcast[[GpuImage]] = Broadcast(name="BatchUploader finished image")
        self._finish_timer: PerformanceTimer = PerformanceTimer(name="Wait upload finished")

    @property
    def dropped(self) -> int:
        """Sets dropped whole because this thread had not taken the ones before them yet."""
        return self._dropped

    def add_gpu_image_callback(self, callback: GpuImageCallback) -> None:
        """Every uploaded image, with its capture time and the event that says it is written, on this thread."""
        self._image_callbacks.add_callback(callback)

    def add_snapshot_callback(self, callback: ImageSnapshotCallback) -> None:
        """Every uploaded set as an `ImageSnapshot`, on this thread, after the image callbacks."""
        self._snapshot_callbacks.add_callback(callback)

    def add_capture_callback(self, callback: CaptureCallback) -> None:
        """Every uploaded set's capture times by cam_id, on this thread, after the snapshot callbacks."""
        self._capture_callbacks.add_callback(callback)

    def add_finished_gpu_image_callback(self, callback: GpuImageCallback) -> None:
        """Every uploaded image, written, on this thread, last."""
        self._finished_image_callbacks.add_callback(callback)

    # ── Lifecycle ─────────────────────────────────────────────────────

    def start(self) -> None:
        with self._wake:
            self._running = True
        self._thread.start()

    def stop(self) -> None:
        """The thread ends; the sets still waiting are dropped and logged."""
        with self._wake:
            self._running = False
            waiting: int = len(self._sets)
            self._sets.clear()
            self._wake.notify()
        if waiting:
            logger.info("BatchUploader: %d sets dropped at stop", waiting)
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
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
        # One stack per size, layout and turn: the four cameras of a rig make one
        stacks: dict[tuple, list[CameraFrame]] = {}
        for frame in frames.values():
            stacks.setdefault((frame.image.shape, frame.image.dtype, frame.nv12, frame.rotate90), []).append(frame)
        images: ImageDict = {}
        ready: torch.cuda.Event | None = None
        for stack in stacks.values():
            uploaded, ready = self._image_upload.upload(stack, self._stream)
            for frame, image in zip(stack, uploaded):
                images[frame.cam_id] = image
        assert ready is not None        # recorded after the last stack, so after every image of the set
        self._timer.add_time((time.perf_counter() - start) * 1000.0, report=False)

        timestamps: dict[int, float] = {cam_id: frame.timestamp for cam_id, frame in frames.items()}
        snapshot: ImageSnapshot = (images, timestamps, ready)
        gpu_images: list[GpuImage] = [GpuImage(cam_id, timestamps[cam_id], image, ready, frames[cam_id])
                                      for cam_id, image in images.items()]
        for gpu_image in gpu_images:
            self._image_callbacks(gpu_image)
        self._snapshot_callbacks(snapshot)
        self._capture_callbacks(timestamps)
        if not self._finished_image_callbacks:
            return
        finish_start: float = time.perf_counter()
        self._stream.synchronize()
        self._finish_timer.add_time((time.perf_counter() - finish_start) * 1000.0, report=False)
        for gpu_image in gpu_images:
            self._finished_image_callbacks(gpu_image)
