"""A YOLO person detector on the host, for the replay backend: the frames of every camera of a round
in one batch, through ONNX Runtime or TensorRT, out as `Detection`s with boxes normalised to the frame.

The frame is stretched whole into the model's input, as the device manip does (nothing cropped,
nobody lost at the edges, boxes map 1:1 onto the frame). A newer frame of a camera replaces one
still waiting: the latest wins, as in the other runners.
"""

import logging
import time
from threading import Event, Lock, Thread
from typing import TYPE_CHECKING, TypeAlias

import cv2
import numpy as np

from modules.utils import PerformanceTimer, Rect

from ..model_types import ModelType
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

        self._shutdown: Event = Event()
        self._wake: Event = Event()
        self._ready: Event = Event()

        self._lock: Lock = Lock()
        self._pending: dict[int, np.ndarray] = {}      # the newest frame per camera, waiting
        self._dropped: int = 0

        self._callback_lock: Lock = Lock()
        self._callbacks: list[DetectionCallback] = []   # run in registration order, on this thread

        self._timer: PerformanceTimer = PerformanceTimer(name="Person Detection  ", sample_count=1000, report_interval=100, color='magenta', omit_init=25)

    @property
    def is_ready(self) -> bool:
        return self._ready.is_set() and not self._shutdown.is_set() and self.is_alive()

    @property
    def dropped(self) -> int:
        """Frames replaced by a newer one before they were looked at."""
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

    def submit(self, cam_id: int, image: np.ndarray) -> None:
        """A frame to look at, on the caller's thread; the newest per camera is kept."""
        if not self.is_ready:
            return
        with self._lock:
            if cam_id in self._pending:
                self._dropped += 1
            self._pending[cam_id] = image
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
        self._runner.infer(np.zeros((1, 3, self.height, self.width), dtype=np.float32))     # warm-up
        self._ready.set()
        logger.info("Person detector: %s ready", self.settings.model)

    def _claim(self) -> dict[int, np.ndarray]:
        with self._lock:
            pending = self._pending
            self._pending = {}
            return pending

    def _process(self) -> None:
        pending: dict[int, np.ndarray] = self._claim()
        if not pending:
            return
        start: float = time.perf_counter()
        cam_ids: list[int] = list(pending.keys())
        batch: np.ndarray = np.stack([self._preprocess(pending[cam_id]) for cam_id in cam_ids])
        output: np.ndarray = self._runner.infer(batch)
        confidence: float = self.settings.confidence
        overlap: float = self.settings.overlap
        detections: list[Detection] = []
        for i, cam_id in enumerate(cam_ids):
            boxes, scores = person_boxes(output[i], confidence, overlap)
            rects: np.ndarray = normalise(boxes, self.width, self.height)
            detections.append(Detection(cam_id=cam_id, boxes=[Rect(*map(float, r)) for r in rects], scores=[float(s) for s in scores]))
        self._timer.add_time((time.perf_counter() - start) * 1000.0, report=self.settings.verbose)
        with self._callback_lock:
            callbacks = list(self._callbacks)
        for detection in detections:
            for callback in callbacks:
                try:
                    callback(detection)
                except Exception:
                    logger.exception("Error in detection callback")

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """BGR or gray, any size, to the model's RGB planar float input, stretched to its size. In C
        order: a transposed array keeps its pixels interleaved, and TensorRT reads the memory as it lies."""
        resized: np.ndarray = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        if resized.ndim == 2:
            rgb: np.ndarray = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        return rgb.transpose(2, 0, 1).astype(np.float32, order='C') / 255.0
