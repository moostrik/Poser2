import logging
import time
from typing import TypeAlias

import numpy as np
import torch

from modules.pose import Frame, Points2D, FrameDictCallbackMixin, replace, FrameDict
from modules.utils import PerformanceTimer

from .. import crop, ModelType
from ..engine import Engine, make_engine
from ..model_worker import ModelWorker
from .decode import decode_simcc
from .settings import PoseEstimatorSettings

logger = logging.getLogger(__name__)

# ImageNet normalization (RGB), for inputs in [0, 1]
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)

Batch: TypeAlias = tuple[FrameDict, list[int], list[torch.Tensor], torch.cuda.Event | None]     # poses, ids, crops, their event


def has_points(frame: Frame) -> bool:
    """Whether a pose carries keypoints: any point with a score."""
    return bool(np.any(frame[Points2D].scores > 0.0))


class PoseEstimator(ModelWorker[Batch, PoseEstimatorSettings], FrameDictCallbackMixin):
    """2D pose points (RTMPose) for the crops of a batch of poses, on its own thread.

    A single-slot queue: a batch still waiting when a newer one comes is dropped and counted in `settings.dropped`;
    only the batches the model ran reach the callbacks, from this thread. With the model off (`enabled`), the poses
    that already carry the camera's own keypoints (an OAK4 pose chain, put on by the frame source) are handed on as
    they are, from the submitting thread, and the others are not posed.
    """

    def __init__(self, settings: PoseEstimatorSettings) -> None:
        ModelWorker.__init__(self, settings, "PoseEstimator")
        FrameDictCallbackMixin.__init__(self)

        self.max_batch: int = min(settings.max_players, 8)
        self._model_size: tuple[int, int] = (settings.width, settings.height)
        self._engine: Engine = make_engine(settings.model_type, f"{settings.model_path}/{settings.model}",
                                           self.max_batch if settings.model_type is ModelType.ONNX else None)
        self._mean: torch.Tensor
        self._std: torch.Tensor

        self._process_timer = PerformanceTimer(name="Pose Estimation", sample_count=1000, report_interval=100, color='yellow', omit_init=25)

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
        self._put(0, (poses, track_ids, images, ready))

    def _process(self, items: dict[int, Batch]) -> None:
        poses, track_ids, images, ready = items[0]
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
        assert self._stream is not None
        self._engine.setup(self._stream)
        dtype: torch.dtype = self._engine.dtype(self._engine.input_names[0])
        with torch.cuda.stream(self._stream):
            self._mean = IMAGENET_MEAN.to(device='cuda', dtype=dtype)
            self._std = IMAGENET_STD.to(device='cuda', dtype=dtype)
            warm_up = torch.zeros((3, self._model_size[1], self._model_size[0]), dtype=torch.float32, device='cuda')
        self._infer([warm_up] * self.max_batch, None)
        width, height = self._model_size
        logger.info(f"Pose: {self.settings.resolution.name} model ready: {width}x{height} {self._engine.precision}")

    def _infer(self, images: list[torch.Tensor], ready: torch.cuda.Event | None) -> tuple[np.ndarray, np.ndarray]:
        """RTMPose's SimCC outputs (N, K, W) and (N, K, H) for the crops (3, h, w) RGB in [0, 1], read once `ready`
        is recorded: the crops resized to the model's input if they differ, normalized as ImageNet, run, and copied
        back to the CPU."""
        assert self._stream is not None
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
