import logging
import time
from threading import Event, Lock, Thread
from typing import TYPE_CHECKING, Callable, TypeAlias

import torch

from modules.pose import FrameDict
from modules.utils import PerformanceTimer

from .. import crop, ModelType
from .runner_onnx import RunnerONNX
from .settings import Settings

if TYPE_CHECKING:
    from .runner_trt import RunnerTRT

logger = logging.getLogger(__name__)

OpticalFlowRunner: TypeAlias = "RunnerONNX | RunnerTRT"

FlowDict: TypeAlias = dict[int, torch.Tensor]
FlowCallback: TypeAlias = Callable[[FlowDict], None]


class Predictor:
    """Dense optical flow (RAFT) from each pose's previous to its current crop, on its own thread, kept on the GPU.

    A single-slot queue: a batch still waiting when a newer one comes is dropped and counted in `settings.dropped`,
    and reaches no callback. Poses without a pair of crops give an empty dict, from the submitting thread; the flows
    come from this thread.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings: Settings = settings

        # One runner per model type; TensorRT is imported only when it is used.
        self._runner: OpticalFlowRunner
        if settings.model_type is ModelType.TRT:
            from .runner_trt import RunnerTRT
            self._runner = RunnerTRT(settings)
        else:
            self._runner = RunnerONNX(settings)

        self._thread: Thread = Thread(target=self._run, daemon=True, name="OpticalFlowPredictor")
        self._shutdown: Event = Event()
        self._wake: Event = Event()
        self._ready: Event = Event()

        self._lock: Lock = Lock()
        self._pending: tuple[list[int], list[tuple[torch.Tensor, torch.Tensor]]] | None = None
        self._pending_timestamp: float = 0.0

        self._callback_lock: Lock = Lock()
        self._callbacks: list[FlowCallback] = []        # run in registration order

        self._process_timer = PerformanceTimer(name="RAFT Optical Flow", sample_count=1000, report_interval=100, color='magenta', omit_init=25)
        self._wait_timer =    PerformanceTimer(name="RAFT Wait        ", sample_count=1000, report_interval=100, color='magenta', omit_init=25)

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
            logger.warning("Optical Flow: predictor thread did not stop cleanly")

    def submit(self, poses: FrameDict, crops: crop.CropImageDict) -> None:
        """The poses and their previous and current crops on the GPU, keyed by track id, to run; the flows go to the
        callbacks."""
        if not self.is_ready or not self.settings.enabled:
            return

        track_ids: list[int] = []
        pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
        for track_id in poses.keys():
            if track_id in crops:
                crop_image = crops[track_id]
                if crop_image.prev_crop is not None:        # both frames needed
                    track_ids.append(track_id)
                    pairs.append((crop_image.prev_crop, crop_image.crop))

        if not pairs:       # no pairs: an empty dict keeps the flow going
            self._notify_callbacks({})
            return

        max_batch: int = self._runner.max_batch
        if len(pairs) > max_batch:
            logger.warning(f"Optical Flow: batch size {len(pairs)} exceeds max {max_batch}, only the first {max_batch} run")
            track_ids, pairs = track_ids[:max_batch], pairs[:max_batch]

        dropped: bool = False
        with self._lock:
            if self._pending is not None:
                dropped = True
                if self.settings.verbose:
                    lag = int((time.perf_counter() - self._pending_timestamp) * 1000)
                    logger.info(f"Optical Flow: dropped a batch with lag {lag} ms")
            self._pending = (track_ids, pairs)
            self._pending_timestamp = time.perf_counter()
        if dropped:
            self.settings.dropped += 1
        self._wake.set()

    def reset(self) -> None:
        """The waiting batch dropped."""
        with self._lock:
            self._pending = None

    def add_flow_callback(self, callback: FlowCallback) -> None:
        with self._callback_lock:
            if callback not in self._callbacks:
                self._callbacks.append(callback)

    def remove_flow_callback(self, callback: FlowCallback) -> None:
        with self._callback_lock:
            if callback in self._callbacks:
                self._callbacks.remove(callback)

    def _notify_callbacks(self, flows: FlowDict) -> None:
        with self._callback_lock:
            callbacks: list[FlowCallback] = list(self._callbacks)
        for callback in callbacks:
            try:
                callback(flows)
            except Exception:
                logger.exception("Optical flow: error in a flow callback")

    def _run(self) -> None:
        try:
            self._runner.setup()
        except Exception:
            logger.exception(f"Optical Flow: failed to load {self._runner.model_file}")
            return
        self._ready.set()
        logger.info(f"Optical Flow: {self.settings.resolution.name} model ready: {self._runner.model_width}x{self._runner.model_height} {self._runner.model_precision}")

        while not self._shutdown.is_set():
            self._wake.wait()
            if self._shutdown.is_set():
                break
            self._wake.clear()
            try:
                self._process()
            except Exception:
                logger.exception("Optical Flow: error processing a batch")

    def _claim(self) -> tuple[list[int], list[tuple[torch.Tensor, torch.Tensor]]] | None:
        with self._lock:
            pending = self._pending
            self._pending = None
            return pending

    def _process(self) -> None:
        pending = self._claim()
        if pending is None:
            return
        track_ids, pairs = pending

        start: float = time.perf_counter()
        flows, lock_wait_ms = self._runner.infer(pairs)
        self._process_timer.add_time((time.perf_counter() - start) * 1000.0 - lock_wait_ms, report=self.settings.verbose)
        self._wait_timer.add_time(lock_wait_ms, report=self.settings.verbose)

        self._notify_callbacks({track_id: flows[i] for i, track_id in enumerate(track_ids)})
