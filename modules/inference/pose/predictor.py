import logging
import time
from threading import Event, Lock, Thread

import numpy as np
import torch

from modules.pose import Frame, Points2D, FrameDictCallbackMixin, replace, FrameDict
from modules.utils import PerformanceTimer

from .. import crop, ModelType
from ..engine import Engine, make_engine
from .decode import decode_simcc
from .settings import PosePredictorSettings

logger = logging.getLogger(__name__)

# ImageNet normalization (RGB), for inputs in [0, 1]
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)


def has_points(frame: Frame) -> bool:
    """Whether a pose carries keypoints: any point with a score."""
    return bool(np.any(frame[Points2D].scores > 0.0))


class Predictor(FrameDictCallbackMixin):
    """2D pose points (RTMPose) for the crops of a batch of poses, on its own thread.

    A single-slot queue: a batch still waiting when a newer one comes is dropped and counted in `settings.dropped`;
    only the batches the model ran reach the callbacks, from this thread. With the model off (`enabled`), the poses
    that already carry the camera's own keypoints (an OAK4 pose chain, put on by the frame source) are handed on as
    they are, from the submitting thread, and the others are not posed.
    """

    def __init__(self, settings: PosePredictorSettings) -> None:
        super().__init__()
        self.settings: PosePredictorSettings = settings

        self.max_batch: int = min(settings.max_players, 8)
        self._model_size: tuple[int, int] = (settings.width, settings.height)
        self._engine: Engine = make_engine(settings.model_type, f"{settings.model_path}/{settings.model}",
                                           self.max_batch if settings.model_type is ModelType.ONNX else None)
        self._stream: torch.cuda.Stream             # made on this thread, in `_run`
        self._mean: torch.Tensor
        self._std: torch.Tensor

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
        if not self.settings.enabled:
            posed: FrameDict = {track_id: frame for track_id, frame in poses.items() if has_points(frame)}
            if posed:
                self._notify_frames_callbacks(posed)
            return
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

        max_batch: int = self.max_batch
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
            self._setup()
        except Exception:
            logger.exception(f"Pose: failed to load {self._engine.model_file}")
            return
        self._ready.set()
        width, height = self._model_size
        logger.info(f"Pose: {self.settings.resolution.name} model ready: {width}x{height} {self._engine.precision}")

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
        simcc_x, simcc_y = self._infer(images, ready)
        keypoints, scores = decode_simcc(simcc_x, simcc_y)
        keypoints[:, :, 0] /= self._model_size[0]
        keypoints[:, :, 1] /= self._model_size[1]
        self._process_timer.add_time((time.perf_counter() - start) * 1000.0, report=self.settings.verbose)

        results: FrameDict = {}
        for i, track_id in enumerate(track_ids):
            if i < len(keypoints) and track_id in poses:
                results[track_id] = replace(poses[track_id], {Points2D: Points2D(values=keypoints[i], scores=scores[i])})
        self._notify_frames_callbacks(results)

    def _setup(self) -> None:
        """The model loaded on this thread, bound to its stream, and run once at its largest batch, so no kernel is
        made while running."""
        self._stream = torch.cuda.Stream()
        self._engine.setup(self._stream)
        dtype: torch.dtype = self._engine.dtype(self._engine.input_names[0])
        with torch.cuda.stream(self._stream):
            self._mean = IMAGENET_MEAN.to(device='cuda', dtype=dtype)
            self._std = IMAGENET_STD.to(device='cuda', dtype=dtype)
            warm_up = torch.zeros((3, self._model_size[1], self._model_size[0]), dtype=torch.float32, device='cuda')
        self._infer([warm_up] * self.max_batch, None)

    def _infer(self, images: list[torch.Tensor], ready: torch.cuda.Event | None) -> tuple[np.ndarray, np.ndarray]:
        """RTMPose's SimCC outputs (N, K, W) and (N, K, H) for the crops (3, h, w) RGB in [0, 1], read once `ready`
        is recorded: the crops resized to the model's input if they differ, normalized as ImageNet, run, and copied
        back to the CPU."""
        width, height = self._model_size
        with torch.cuda.stream(self._stream):
            if ready is not None:
                self._stream.wait_event(ready)      # the crops are written on the crop stream
            batch: torch.Tensor = torch.stack(images, dim=0).to(self._mean.dtype)
            if batch.shape[2] != height or batch.shape[3] != width:
                batch = torch.nn.functional.interpolate(batch, size=(height, width), mode='bilinear', align_corners=False)
            batch = (batch - self._mean) / self._std
            outputs: dict[str, torch.Tensor] = self._engine.run({self._engine.input_names[0]: batch})
        self._stream.synchronize()
        simcc_x, simcc_y = (outputs[name] for name in self._engine.output_names)
        return simcc_x.float().cpu().numpy(), simcc_y.float().cpu().numpy()
