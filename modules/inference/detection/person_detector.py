"""A YOLO person detector on the host, for the replay backend: the uploaded images of every camera of a
round in one batch, through ONNX Runtime or TensorRT, out as `Detection`s with boxes normalised to the frame.

It takes the `Uploader`'s images (`GpuImage`), so a frame goes to the GPU once and nothing of it is worked
on the CPU: each image is stretched whole into the model's input on the GPU, as the device manip does
(nothing cropped, nobody lost at the edges, boxes map 1:1 onto the frame), and only the person rows
come back. The image is straightened and mirrored, so are its boxes. A newer image of a camera replaces
one still waiting: the latest wins, as in the other runners.
"""

import logging
import time
from threading import Event, Lock, Thread
from typing import TYPE_CHECKING, TypeAlias

import numpy as np
import torch
import torch.nn.functional as F

from modules.utils import PerformanceTimer, Rect

from ..model_types import ModelType
from ..source import GpuImage
from .decode import normalise, person_boxes
from .io import Detection, DetectionCallback
from .runner_onnx import RunnerONNX
from .settings import Settings

if TYPE_CHECKING:
    from .runner_trt import RunnerTRT

logger = logging.getLogger(__name__)

DetectionRunner: TypeAlias = "RunnerONNX | RunnerTRT"


class PersonDetector(Thread):
    def __init__(self, settings: Settings) -> None:
        super().__init__(daemon=True)
        self.settings: Settings = settings
        self.width: int = settings.width
        self.height: int = settings.height

        # One runner per model type; TensorRT is imported only when it is used.
        self._runner: DetectionRunner
        if settings.model_type is ModelType.ONNX:
            self._runner = RunnerONNX(settings.model_file)
        else:
            from .runner_trt import RunnerTRT
            self._runner = RunnerTRT(settings.model_file)
        self._stream: torch.cuda.Stream | None = None      # made on this thread, in `_setup`

        self._shutdown: Event = Event()
        self._wake: Event = Event()
        self._ready: Event = Event()

        self._lock: Lock = Lock()
        self._pending: dict[int, GpuImage] = {}     # the newest image per camera, waiting
        self._dropped: int = 0

        self._callback_lock: Lock = Lock()
        self._callbacks: list[DetectionCallback] = []   # run in registration order, on this thread

        self._timer: PerformanceTimer = PerformanceTimer(name="Person Detection  ", sample_count=1000, report_interval=100, color='magenta', omit_init=25)

    @property
    def is_ready(self) -> bool:
        return self._ready.is_set() and not self._shutdown.is_set() and self.is_alive()

    @property
    def dropped(self) -> int:
        """Images replaced by a newer one before they were looked at."""
        return self._dropped

    def stop(self) -> None:
        self._shutdown.set()
        self._wake.set()
        if self.is_alive():
            self.join(timeout=2.0)
        if self.is_alive():
            logger.warning("Person detector thread did not stop cleanly")

    def add_detection_callback(self, callback: DetectionCallback) -> None:
        with self._callback_lock:
            if callback not in self._callbacks:
                self._callbacks.append(callback)

    @property
    def enabled(self) -> bool:
        """Whether it looks for players: off, a replay plays the recording's own tracklets."""
        return self.settings.enabled

    def submit(self, image: GpuImage) -> None:
        """An uploaded image to look at (`Uploader`'s callback, the camera's thread); the newest per camera is kept.
        Disabled, it looks at nothing."""
        if not self.is_ready or not self.settings.enabled:
            return
        with self._lock:
            if image.cam_id in self._pending:
                self._dropped += 1
            self._pending[image.cam_id] = image
        self._wake.set()

    def run(self) -> None:
        try:
            self._setup()
        except Exception as e:
            logger.error("Person detector: failed to load %s: %s", self.settings.model_file, e)
            return
        while not self._shutdown.is_set():
            self._wake.wait()
            if self._shutdown.is_set():
                break
            self._wake.clear()
            try:
                self._process()
            except Exception:
                logger.exception("Person detector error")

    def _setup(self) -> None:
        self._runner.setup()
        self._stream = torch.cuda.Stream()
        with torch.cuda.stream(self._stream):
            warm_up: torch.Tensor = torch.zeros((1, 3, self.height, self.width), dtype=torch.float32, device='cuda')
        self._runner.infer(warm_up, self._stream)
        self._ready.set()
        logger.info("Person detector: %s ready", self.settings.model)

    def _claim(self) -> dict[int, GpuImage]:
        with self._lock:
            pending = self._pending
            self._pending = {}
            return pending

    def _process(self) -> None:
        """The pending images as one batch. They are held until the runner has synchronized its stream,
        so the upload's memory is not reused while this stream reads it."""
        pending: dict[int, GpuImage] = self._claim()
        if not pending:
            return
        assert self._stream is not None
        start: float = time.perf_counter()
        cam_ids: list[int] = list(pending.keys())
        batch: torch.Tensor = self._batch([pending[cam_id] for cam_id in cam_ids])
        output: np.ndarray = self._runner.infer(batch, self._stream)
        confidence: float = self.settings.confidence
        overlap: float = self.settings.overlap
        detections: list[Detection] = []
        for i, cam_id in enumerate(cam_ids):
            boxes, scores = person_boxes(output[i], confidence, overlap)
            rects: np.ndarray = normalise(boxes, self.width, self.height)
            detections.append(Detection(cam_id=cam_id, timestamp=pending[cam_id].timestamp, boxes=[Rect(*map(float, r)) for r in rects],
                                        scores=[float(s) for s in scores]))
        self._timer.add_time((time.perf_counter() - start) * 1000.0, report=self.settings.verbose)
        with self._callback_lock:
            callbacks = list(self._callbacks)
        for detection in detections:
            for callback in callbacks:
                try:
                    callback(detection)
                except Exception:
                    logger.exception("Error in detection callback")

    def _batch(self, images: list[GpuImage]) -> torch.Tensor:
        """The images stretched to the model's size, (N, 3, height, width) float32 RGB in [0, 1], on this
        detector's stream once each image's upload is written."""
        assert self._stream is not None
        with torch.cuda.stream(self._stream):
            resized: list[torch.Tensor] = []
            for image in images:
                self._stream.wait_event(image.ready)
                resized.append(F.interpolate(image.image.unsqueeze(0).float(), size=(self.height, self.width),
                                             mode='bilinear', align_corners=False)[0])
            return torch.stack(resized)
