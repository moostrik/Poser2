import logging
import os
import time
from typing import TypeAlias

import torch

from modules.pose import FrameDict
from modules.utils import Broadcast, PerformanceTimer, Rect

from ..engine import Engine, ModelType, ModelWorker, make_engine
from ..preprocessing import crop
from .image import Image, ImageDict, ImageCallback
from .recurrent_state import RecurrentState
from .settings import SegmenterSettings

logger = logging.getLogger(__name__)

RECURRENT_CHANNELS: tuple[tuple[int, int], ...] = ((16, 2), (20, 4), (40, 8), (64, 16))     # (channels, downscale) of r1..r4
STATE_INPUTS: tuple[str, ...] = ('r1i', 'r2i', 'r3i', 'r4i')
STATE_OUTPUTS: tuple[str, ...] = ('r1o', 'r2o', 'r3o', 'r4o')

Batch: TypeAlias = tuple[FrameDict, list[int], list[torch.Tensor], list[Rect | None], torch.cuda.Event | None]     # poses, ids, crops, their boxes, their event


class Segmenter(ModelWorker[Batch, SegmenterSettings]):
    """Person masks (Robust Video Matting) for the crops of a batch of poses, on its own thread, kept on the GPU.

    A single-slot queue: a batch still waiting when a newer one comes is dropped and counted in `settings.dropped`.
    A dropped batch reaches no callback. Poses without crops go to the callbacks without masks, from the submitting thread;
    the masks from this thread.
    """

    def __init__(self, settings: SegmenterSettings) -> None:
        super().__init__(settings, "Segmenter")

        self.max_batch: int = min(settings.max_players, 4)
        self._model_size: tuple[int, int] = (settings.width, settings.height)
        self._engine: Engine = make_engine(settings.model_type, settings.model_file(settings.model),
                                           self.max_batch if settings.model_type is ModelType.ONNX else None)
        self._states: dict[int, RecurrentState] = {}    # per track id: each player's memory of the earlier frames
        self._zero_states: list[torch.Tensor] = []      # (1, C, h, w) per recurrent input, for a new player
        self._reset_interval: int = settings.reset_interval
        self._batch_count: int = 0

        self._callbacks: Broadcast[[FrameDict, ImageDict]] = Broadcast(name="Segmentation mask")

        self._process_timer = PerformanceTimer(name="Segmentation", sample_count=1000, report_interval=100, color='cyan', omit_init=25)

    def submit(self, poses: FrameDict, crops: crop.CropImageDict) -> None:
        """The poses and their crops on the GPU, keyed by track id, to run; the results go to the callbacks."""
        if not self.is_ready or not self.settings.enabled:
            return

        track_ids: list[int] = []
        images: list[torch.Tensor] = []
        boxes: list[Rect | None] = []
        ready: torch.cuda.Event | None = None      # the crops' event, one for the batch
        for track_id in poses.keys():
            if track_id in crops:
                track_ids.append(track_id)
                images.append(crops[track_id].crop)
                boxes.append(crops[track_id].box)
                ready = crops[track_id].ready

        if not images:      # no crops: the poses go on without masks
            self._notify_callbacks(poses, {})
            return

        max_batch: int = self.max_batch
        if len(images) > max_batch:
            logger.warning(f"Segmentation: batch size {len(images)} exceeds max {max_batch}, only the first {max_batch} run")
            track_ids, images, boxes = track_ids[:max_batch], images[:max_batch], boxes[:max_batch]
        self._put(0, (poses, track_ids, images, boxes, ready))

    def add_segmentation_callback(self, callback: ImageCallback) -> None:
        self._callbacks.add_callback(callback)

    def remove_segmentation_callback(self, callback: ImageCallback) -> None:
        self._callbacks.remove_callback(callback)

    def _notify_callbacks(self, poses: FrameDict, images: ImageDict) -> None:
        self._callbacks(poses, images)

    def _process(self, items: dict[int, Batch]) -> None:
        poses, track_ids, images, boxes, ready = items[0]

        # Every player's memory cleared each `reset_interval` batches
        self._batch_count += 1
        if self._reset_interval > 0 and self._batch_count % self._reset_interval == 0:
            self._states.clear()

        start: float = time.perf_counter()
        masks, foregrounds, written = self._infer(images, track_ids, ready)
        self._process_timer.add_time((time.perf_counter() - start) * 1000.0, report=self.settings.verbose)

        results: ImageDict = {}
        for i, track_id in enumerate(track_ids):
            if track_id in poses:
                results[track_id] = Image(mask=masks[i], foreground=foregrounds[i], timestamp=poses[track_id].timestamp,
                                           ready=written, box=boxes[i])
        self._notify_callbacks(poses, results)

    def _setup(self) -> None:
        assert self._stream is not None
        self._engine.setup(self._stream)
        width, height = self._model_size
        dtype: torch.dtype = self._engine.dtype('src')
        with torch.cuda.stream(self._stream):
            self._zero_states = [torch.zeros((1, c, height // s, width // s), dtype=dtype, device='cuda') for c, s in RECURRENT_CHANNELS]
            warm_up = torch.zeros((3, height, width), dtype=torch.float32, device='cuda')
        self._infer([warm_up] * self.max_batch, list(range(self.max_batch)), None)
        self._stream.synchronize()
        self._states.clear()
        logger.info(f"Segmentation: {self.settings.resolution.name} model ready: {width}x{height} {os.path.basename(self._engine.model_file)}")

    def _infer(self, images: list[torch.Tensor], track_ids: list[int],
               ready: torch.cuda.Event | None) -> tuple[torch.Tensor, torch.Tensor, torch.cuda.Event]:
        """The alpha matte (N, h, w) and the foreground (N, 3, h, w), the crops multiplied by it, for the crops (3, h, w)
        RGB in [0, 1] of the players `track_ids`, read once `ready` is recorded; and the event recorded after they are
        written. Each player's recurrent state goes in with its crop and comes out for its next."""
        assert self._stream is not None
        width, height = self._model_size
        src_dtype: torch.dtype = self._engine.dtype('src')
        with torch.cuda.stream(self._stream):
            if ready is not None:
                self._stream.wait_event(ready)      # the crops are written on the crop stream
            original: torch.Tensor = torch.stack(images, dim=0).to(src_dtype)
            crop_h, crop_w = original.shape[2], original.shape[3]
            needs_resize: bool = crop_h != height or crop_w != width
            src: torch.Tensor = original
            if needs_resize:
                src = torch.nn.functional.interpolate(original, size=(height, width), mode='bilinear', align_corners=False)
            inputs: dict[str, torch.Tensor] = {'src': src}
            for k, name in enumerate(STATE_INPUTS):
                sources = [self._states[t].tensors()[k] if t in self._states else self._zero_states[k] for t in track_ids]
                inputs[name] = torch.cat(sources, dim=0)
            outputs: dict[str, torch.Tensor] = self._engine.run(inputs)
            self._states = {t: RecurrentState(*(outputs[name][i:i + 1] for name in STATE_OUTPUTS)) for i, t in enumerate(track_ids)}
            mask: torch.Tensor = outputs['pha'].squeeze(1)
            if needs_resize:
                mask = torch.nn.functional.interpolate(mask.unsqueeze(1), size=(crop_h, crop_w), mode='bilinear', align_corners=False).squeeze(1)
            foreground: torch.Tensor = original * mask.unsqueeze(1)
            written: torch.cuda.Event = torch.cuda.Event()
            written.record(self._stream)
        self._stream.synchronize()              # handed on written: its readers are outside the package
        return mask, foreground, written
