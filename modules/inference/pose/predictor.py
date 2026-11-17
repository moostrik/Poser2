import logging
import time
from threading import Event, Lock, Thread
from typing import TYPE_CHECKING, TypeAlias

import torch

from modules.pose import Points2D, FrameDictCallbackMixin, replace, FrameDict
from modules.utils import PerformanceTimer

from .. import crop, ModelType
from .decode import decode_simcc
from .runner_onnx import RunnerONNX
from .settings import Settings

if TYPE_CHECKING:
    from .runner_trt import RunnerTRT

logger = logging.getLogger(__name__)

PoseRunner: TypeAlias = "RunnerONNX | RunnerTRT"


class Predictor(FrameDictCallbackMixin):
    """2D pose points (RTMPose) for the crops of a batch of poses, on its own thread.

    A single-slot queue: a batch still waiting when a newer one comes is dropped and counted in `settings.dropped`;
    only the batches the model ran reach the callbacks, from this thread.
    """

    def __init__(self, settings: Settings) -> None:
        super().__init__()
        self.settings: Settings = settings

        # One runner per model type; TensorRT is imported only when it is used.
        self._runner: PoseRunner
        if settings.model_type is ModelType.ONNX:
            self._runner = RunnerONNX(settings)
        else:
            from .runner_trt import RunnerTRT
            self._runner = RunnerTRT(settings)

        self._thread: Thread = Thread(target=self._run, daemon=True, name="PosePredictor")
        self._shutdown: Event = Event()
        self._wake: Event = Event()
        self._ready: Event = Event()

        self._lock: Lock = Lock()
        self._pending: tuple[FrameDict, list[int], list[torch.Tensor], torch.cuda.Event | None] | None = None
        self._pending_timestamp: float = 0.0

        self._process_timer = PerformanceTimer(name="RTM Pose Detection", sample_count=1000, report_interval=100, color='yellow', omit_init=25)

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
            logger.warning("Pose: predictor thread did not stop cleanly")

    def submit(self, poses: FrameDict, crops: crop.CropImageDict) -> None:
        """The poses and their crops on the GPU, keyed by track id, to run; the results go to the callbacks."""
        if not self.is_ready:
            return

        track_ids: list[int] = []
        images: list[torch.Tensor] = []
        ready: torch.cuda.Event | None = None      # the crops' event, one for the batch
        for track_id in poses.keys():
            if track_id in crops:
                track_ids.append(track_id)
                images.append(crops[track_id].crop)
                ready = crops[track_id].ready

        max_batch: int = self._runner.max_batch
        if len(images) > max_batch:
            logger.warning(f"Pose: batch size {len(images)} exceeds max {max_batch}, only the first {max_batch} run")
            track_ids, images = track_ids[:max_batch], images[:max_batch]

        dropped: bool = False
        with self._lock:
            if self._pending is not None:
                dropped = True
                if self.settings.verbose:
                    lag = int((time.perf_counter() - self._pending_timestamp) * 1000)
                    logger.info(f"Pose: dropped a batch with lag {lag} ms")
            self._pending = (poses, track_ids, images, ready)
            self._pending_timestamp = time.perf_counter()
        if dropped:
            self.settings.dropped += 1
        self._wake.set()

    def reset(self) -> None:
        """The waiting batch dropped."""
        with self._lock:
            self._pending = None

    def _run(self) -> None:
        try:
            self._runner.setup()
        except Exception:
            logger.exception(f"Pose: failed to load {self._runner.model_file}")
            return
        self._ready.set()
        logger.info(f"Pose: {self.settings.resolution.name} model ready: {self._runner.model_width}x{self._runner.model_height} {self._runner.model_precision}")

        while not self._shutdown.is_set():
            self._wake.wait()
            if self._shutdown.is_set():
                break
            self._wake.clear()
            try:
                self._process()
            except Exception:
                logger.exception("Pose: error processing a batch")

    def _claim(self) -> tuple[FrameDict, list[int], list[torch.Tensor], torch.cuda.Event | None] | None:
        with self._lock:
            pending = self._pending
            self._pending = None
            return pending

    def _process(self) -> None:
        pending = self._claim()
        if pending is None:
            return
        poses, track_ids, images, ready = pending
        if not images:
            self._notify_frames_callbacks({})
            return

        start: float = time.perf_counter()
        simcc_x, simcc_y, lock_wait_ms = self._runner.infer(images, ready)
        keypoints, scores = decode_simcc(simcc_x, simcc_y)
        keypoints[:, :, 0] /= self._runner.model_width
        keypoints[:, :, 1] /= self._runner.model_height
        self._process_timer.add_time((time.perf_counter() - start) * 1000.0 - lock_wait_ms, report=self.settings.verbose)

        results: FrameDict = {}
        for i, track_id in enumerate(track_ids):
            if i < len(keypoints) and track_id in poses:
                results[track_id] = replace(poses[track_id], {Points2D: Points2D(values=keypoints[i], scores=scores[i])})
        self._notify_frames_callbacks(results)
