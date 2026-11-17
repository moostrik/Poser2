import logging
import time
from threading import Event, Lock, Thread
from typing import TYPE_CHECKING, TypeAlias

import torch

from modules.pose import FrameDict
from modules.utils import PerformanceTimer

from .. import crop, ModelType
from .image import Image, ImageDict, ImageCallback
from .runner_onnx import RunnerONNX
from .settings import Settings

if TYPE_CHECKING:
    from .runner_trt import RunnerTRT

logger = logging.getLogger(__name__)

SegmentationRunner: TypeAlias = "RunnerONNX | RunnerTRT"


class Predictor:
    """Person masks (Robust Video Matting) for the crops of a batch of poses, on its own thread, kept on the GPU.

    A single-slot queue: a batch still waiting when a newer one comes is dropped and counted in `settings.dropped`.
    A dropped batch reaches no callback: a newer one is already waiting, and an empty one would read as no player and
    blank the masks until it arrives. Poses without crops go to the callbacks without masks, from the submitting thread;
    the masks from this thread.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings: Settings = settings

        # One runner per model type; TensorRT is imported only when it is used.
        self._runner: SegmentationRunner
        if settings.model_type is ModelType.TRT:
            from .runner_trt import RunnerTRT
            self._runner = RunnerTRT(settings)
        else:
            self._runner = RunnerONNX(settings)
        self._reset_interval: int = settings.reset_interval
        self._batch_count: int = 0

        self._thread: Thread = Thread(target=self._run, daemon=True, name="SegmentationPredictor")
        self._shutdown: Event = Event()
        self._wake: Event = Event()
        self._ready: Event = Event()

        self._lock: Lock = Lock()
        self._pending: tuple[FrameDict, list[int], list[torch.Tensor]] | None = None
        self._pending_timestamp: float = 0.0

        self._callback_lock: Lock = Lock()
        self._callbacks: list[ImageCallback] = []       # run in registration order

        self._process_timer = PerformanceTimer(name="RVM Segmentation  ", sample_count=1000, report_interval=100, color='cyan', omit_init=25)

    @property
    def is_ready(self) -> bool:
        return self._ready.is_set() and not self._shutdown.is_set() and self._thread.is_alive()

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._shutdown.set()
        self._wake.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        if self._thread.is_alive():
            logger.warning("Segmentation: predictor thread did not stop cleanly")

    def submit(self, poses: FrameDict, crops: crop.CropImageDict) -> None:
        """The poses and their crops on the GPU, keyed by track id, to run; the results go to the callbacks."""
        if not self.is_ready or not self.settings.enabled:
            return

        track_ids: list[int] = []
        images: list[torch.Tensor] = []
        for track_id in poses.keys():
            if track_id in crops:
                track_ids.append(track_id)
                images.append(crops[track_id].crop)

        if not images:      # no crops: the poses go on without masks
            self._notify_callbacks(poses, {})
            return

        max_batch: int = self._runner.max_batch
        if len(images) > max_batch:
            logger.warning(f"Segmentation: batch size {len(images)} exceeds max {max_batch}, only the first {max_batch} run")
            track_ids, images = track_ids[:max_batch], images[:max_batch]

        dropped: bool = False
        with self._lock:
            if self._pending is not None:
                dropped = True
                if self.settings.verbose:
                    lag = int((time.perf_counter() - self._pending_timestamp) * 1000)
                    logger.info(f"Segmentation: dropped a batch with lag {lag} ms")
            self._pending = (poses, track_ids, images)
            self._pending_timestamp = time.perf_counter()
        if dropped:
            self.settings.dropped += 1
        self._wake.set()

    def reset(self) -> None:
        """The waiting batch dropped."""
        with self._lock:
            self._pending = None

    def add_segmentation_callback(self, callback: ImageCallback) -> None:
        with self._callback_lock:
            if callback not in self._callbacks:
                self._callbacks.append(callback)

    def remove_segmentation_callback(self, callback: ImageCallback) -> None:
        with self._callback_lock:
            if callback in self._callbacks:
                self._callbacks.remove(callback)

    def _notify_callbacks(self, poses: FrameDict, images: ImageDict) -> None:
        with self._callback_lock:
            callbacks: list[ImageCallback] = list(self._callbacks)
        for callback in callbacks:
            try:
                callback(poses, images)
            except Exception:
                logger.exception("Segmentation: error in a mask callback")

    def _run(self) -> None:
        try:
            self._runner.setup()
        except Exception:
            logger.exception(f"Segmentation: failed to load {self._runner.model_file}")
            return
        self._ready.set()
        logger.info(f"Segmentation: {self.settings.resolution.name} model ready: {self._runner.model_width}x{self._runner.model_height} {self._runner.model_precision}")

        while not self._shutdown.is_set():
            self._wake.wait()
            if self._shutdown.is_set():
                break
            self._wake.clear()
            try:
                self._process()
            except Exception:
                logger.exception("Segmentation: error processing a batch")

    def _claim(self) -> tuple[FrameDict, list[int], list[torch.Tensor]] | None:
        with self._lock:
            pending = self._pending
            self._pending = None
            return pending

    def _process(self) -> None:
        pending = self._claim()
        if pending is None:
            return
        poses, track_ids, images = pending

        # Every player's memory cleared now and then, a failsafe against a state gone wrong
        self._batch_count += 1
        if self._reset_interval > 0 and self._batch_count % self._reset_interval == 0:
            self._runner.reset_states()

        start: float = time.perf_counter()
        masks, foregrounds, lock_wait_ms = self._runner.infer(images, track_ids)
        self._process_timer.add_time((time.perf_counter() - start) * 1000.0 - lock_wait_ms, report=self.settings.verbose)

        results: ImageDict = {}
        for i, track_id in enumerate(track_ids):
            if track_id in poses:
                results[track_id] = Image(mask=masks[i], foreground=foregrounds[i], timestamp=poses[track_id].timestamp)
        self._notify_callbacks(poses, results)
