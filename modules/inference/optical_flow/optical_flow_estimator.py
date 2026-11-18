import logging
import time
from threading import Lock
from typing import TypeAlias

import torch

from modules.pose import FrameDict
from modules.utils import PerformanceTimer

from ..engine import Engine, ModelType, ModelWorker, make_engine
from ..preprocessing import crop
from .flow import FlowDict, FlowCallback
from .settings import OpticalFlowEstimatorSettings

logger = logging.getLogger(__name__)

Batch: TypeAlias = tuple[list[int], list[tuple[torch.Tensor, torch.Tensor]], torch.cuda.Event | None]    # ids, crop pairs, their event


class OpticalFlowEstimator(ModelWorker[Batch, OpticalFlowEstimatorSettings]):
    """Dense optical flow (RAFT) from each pose's previous to its current crop, on its own thread, kept on the GPU.

    A single-slot queue: a batch still waiting when a newer one comes is dropped and counted in `settings.dropped`,
    and reaches no callback. Poses without a pair of crops give an empty dict, from the submitting thread; the flows
    come from this thread.
    """

    def __init__(self, settings: OpticalFlowEstimatorSettings) -> None:
        super().__init__(settings, "OpticalFlowEstimator")

        onnx: bool = settings.model_type is ModelType.ONNX
        self.max_batch: int = min(settings.max_players, 8 if onnx else 4)
        self._model_size: tuple[int, int] = (settings.width, settings.height)
        self._engine: Engine = make_engine(settings.model_type, f"{settings.model_path}/{settings.model}",
                                           self.max_batch if onnx else None)

        self._callback_lock: Lock = Lock()
        self._callbacks: list[FlowCallback] = []        # run in registration order

        self._process_timer = PerformanceTimer(name="Optical Flow Estimation", sample_count=1000, report_interval=100, color='magenta', omit_init=25)

    def submit(self, poses: FrameDict, crops: crop.CropImageDict) -> None:
        """The poses and their previous and current crops on the GPU, keyed by track id, to run; the flows go to the
        callbacks."""
        if not self.is_ready or not self.settings.enabled:
            return

        track_ids: list[int] = []
        pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
        ready: torch.cuda.Event | None = None      # the crops' event, one for the batch
        for track_id in poses.keys():
            if track_id in crops:
                crop_image = crops[track_id]
                if crop_image.prev_crop is not None:        # both frames needed
                    track_ids.append(track_id)
                    pairs.append((crop_image.prev_crop, crop_image.crop))
                    ready = crop_image.ready

        if not pairs:       # no pairs: an empty dict keeps the flow going
            self._notify_callbacks({})
            return

        max_batch: int = self.max_batch
        if len(pairs) > max_batch:
            logger.warning(f"Optical Flow: batch size {len(pairs)} exceeds max {max_batch}, only the first {max_batch} run")
            track_ids, pairs = track_ids[:max_batch], pairs[:max_batch]
        self._put(0, (track_ids, pairs, ready))

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

    def _process(self, items: dict[int, Batch]) -> None:
        track_ids, pairs, ready = items[0]

        start: float = time.perf_counter()
        flows = self._infer(pairs, ready)
        self._process_timer.add_time((time.perf_counter() - start) * 1000.0, report=self.settings.verbose)

        self._notify_callbacks({track_id: flows[i] for i, track_id in enumerate(track_ids)})

    def _setup(self) -> None:
        assert self._stream is not None
        self._engine.setup(self._stream)
        width, height = self._model_size
        with torch.cuda.stream(self._stream):
            warm_up = torch.zeros((3, height, width), dtype=torch.float32, device='cuda')
        self._infer([(warm_up, warm_up)] * self.max_batch, None)
        self._stream.synchronize()
        logger.info(f"Optical Flow: {self.settings.resolution.name} model ready: {width}x{height} {self._engine.precision}")

    def _infer(self, pairs: list[tuple[torch.Tensor, torch.Tensor]], ready: torch.cuda.Event | None) -> torch.Tensor:
        """The flow (N, 2, H, W), x and y in the model's pixels, from each previous to current crop (3, h, w) RGB in
        [0, 1], read once `ready` is recorded: RAFT reads 0..255 at its input size."""
        assert self._stream is not None
        width, height = self._model_size
        first, second = self._engine.input_names[0], self._engine.input_names[1]
        dtype: torch.dtype = self._engine.dtype(first)
        with torch.cuda.stream(self._stream):
            if ready is not None:
                self._stream.wait_event(ready)      # the crops are written on the crop stream
            previous: torch.Tensor = torch.stack([p[0] for p in pairs], dim=0).mul(255.0).to(dtype)
            current: torch.Tensor = torch.stack([p[1] for p in pairs], dim=0).mul(255.0).to(dtype)
            if previous.shape[2] != height or previous.shape[3] != width:
                previous = torch.nn.functional.interpolate(previous, size=(height, width), mode='bilinear', align_corners=False)
                current = torch.nn.functional.interpolate(current, size=(height, width), mode='bilinear', align_corners=False)
            flows: torch.Tensor = self._engine.run({first: previous, second: current})[self._engine.output_names[0]]
        self._stream.synchronize()
        return flows
