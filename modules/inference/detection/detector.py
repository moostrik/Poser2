"""A YOLO person detector on the host, for the replay backend: the frames of every camera of a round
in one batch, through ONNX Runtime, out as `Detection`s with boxes normalised to the frame.

The frame is stretched whole into the model's input, as the device manip does (nothing cropped,
nobody lost at the edges, boxes map 1:1 onto the frame). A newer frame of a camera replaces one
still waiting: the latest wins, as in the other runners.
"""

import json
import logging
import time
from threading import Event, Lock, Thread

import cv2
import numpy as np
import onnxruntime as ort

from modules.utils import PerformanceTimer, Rect

from ..ort_runtime import preload_cuda
from .decode import normalise, person_boxes
from .io import Detection, DetectionCallback
from .settings import Settings

logger = logging.getLogger(__name__)


class Detector(Thread):
    def __init__(self, settings: Settings) -> None:
        super().__init__(daemon=True)
        self.settings: Settings = settings
        self.width: int = 0             # the model's input, read from the file at setup
        self.height: int = 0

        self._shutdown: Event = Event()
        self._wake: Event = Event()
        self._ready: Event = Event()

        self._lock: Lock = Lock()
        self._pending: dict[int, np.ndarray] = {}      # the newest frame per camera, waiting
        self._dropped: int = 0

        self._callback_lock: Lock = Lock()
        self._callbacks: list[DetectionCallback] = []   # run in registration order, on this thread

        self._session: ort.InferenceSession | None = None
        self._input_name: str = ''
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

    def register_callback(self, callback: DetectionCallback) -> None:
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
        preload_cuda()
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.log_severity_level = 3
        self._session = ort.InferenceSession(self.settings.model_file, options, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        provider: str = self._session.get_providers()[0]
        if provider != 'CUDAExecutionProvider':
            logger.warning("Person detector runs on %s: CUDA is not available to ONNX Runtime, expect dropped frames", provider)
        model_input = self._session.get_inputs()[0]
        self._input_name = model_input.name
        self.height, self.width = self._input_size(model_input.shape)
        self._infer(np.zeros((1, 3, self.height, self.width), dtype=np.float32))     # warm-up
        self._ready.set()
        logger.info("Person detector: %s ready, %dx%d on %s", self.settings.model, self.width, self.height, provider)

    def _input_size(self, shape: list) -> tuple[int, int]:
        """The model's input height and width: from its shape (batch, 3, height, width) when the
        file fixes them, else from the `imgsz` the export wrote into the metadata, since
        ultralytics exports every axis but the channels as a symbol."""
        if isinstance(shape[2], int) and isinstance(shape[3], int):
            return shape[2], shape[3]
        assert self._session is not None
        imgsz: str | None = self._session.get_modelmeta().custom_metadata_map.get('imgsz')
        if imgsz is None:
            raise ValueError(f"input shape {shape} has no fixed size and the model carries no imgsz metadata")
        height, width = json.loads(imgsz)
        return int(height), int(width)

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
        output: np.ndarray = self._infer(batch)
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
        """BGR or gray, any size, to the model's RGB planar float input, stretched to its size."""
        resized: np.ndarray = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        if resized.ndim == 2:
            rgb: np.ndarray = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        return rgb.transpose(2, 0, 1).astype(np.float32) / 255.0

    def _infer(self, batch: np.ndarray) -> np.ndarray:
        assert self._session is not None
        return self._session.run(None, {self._input_name: batch})[0]
