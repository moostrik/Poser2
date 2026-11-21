"""A YOLO person detector on the host, for the replay backend: the uploaded images of every camera of a
round in one batch, through ONNX Runtime or TensorRT, out as `Detection`s with boxes normalized to the frame.

It takes the uploaded images (`GpuImage`), so a frame goes to the GPU once and nothing of it is worked
on the CPU: each image is stretched whole into the model's input on the GPU, as the device manip does, and
only the person rows come back. The image is straightened and mirrored, so are its boxes. A newer image of a camera replaces
one still waiting: the latest wins, as in the other workers.
"""

import logging
import time

import numpy as np
import torch
import torch.nn.functional as F

from modules.utils import Broadcast, PerformanceTimer, Rect

from ..engine import Engine, ModelWorker, make_engine
from ..preprocessing import GpuImage
from .decode import PERSON_ROWS, normalize, person_boxes
from .detection import Detection, DetectionCallback
from .settings import PersonDetectorSettings

logger = logging.getLogger(__name__)


class PersonDetector(ModelWorker[GpuImage, PersonDetectorSettings]):
    def __init__(self, settings: PersonDetectorSettings) -> None:
        super().__init__(settings, "PersonDetector")
        self.width: int = settings.width
        self.height: int = settings.height
        self._engine: Engine = make_engine(settings.model_type, settings.model_file(settings.model))

        self._callbacks: Broadcast[Detection] = Broadcast(name="detection")

        self._timer: PerformanceTimer = PerformanceTimer(name="Person Detection", sample_count=1000, report_interval=100, color='magenta', omit_init=25)

    def add_detection_callback(self, callback: DetectionCallback) -> None:
        self._callbacks.add_callback(callback)

    @property
    def enabled(self) -> bool:
        """Whether it looks for players: off, a replay plays the recording's own tracklets."""
        return self.settings.enabled

    def submit(self, image: GpuImage) -> None:
        """An uploaded image to look at (an uploader's image callback, the uploader's thread); the newest per camera is kept.
        Disabled, it looks at nothing."""
        if not self.is_ready or not self.settings.enabled:
            return
        self._put(image.cam_id, image)

    def _setup(self) -> None:
        assert self._stream is not None
        self._engine.setup(self._stream)
        with torch.cuda.stream(self._stream):
            warm_up: torch.Tensor = torch.zeros((1, 3, self.height, self.width), dtype=torch.float32, device='cuda')
        self._infer(warm_up)
        logger.info("Person detector: %s ready", self.settings.model)

    def _process(self, items: dict[int, GpuImage]) -> None:
        """The waiting images, one per camera, as one batch. They are held until the model has synchronized its
        stream."""
        start: float = time.perf_counter()
        cam_ids: list[int] = list(items.keys())
        batch: torch.Tensor = self._batch([items[cam_id] for cam_id in cam_ids])
        output: np.ndarray = self._infer(batch)
        confidence: float = self.settings.confidence
        overlap: float = self.settings.overlap
        detections: list[Detection] = []
        for i, cam_id in enumerate(cam_ids):
            boxes, scores = person_boxes(output[i], confidence, overlap)
            rects: np.ndarray = normalize(boxes, self.width, self.height)
            detections.append(Detection(cam_id=cam_id, timestamp=items[cam_id].timestamp, boxes=[Rect(*map(float, r)) for r in rects],
                                        scores=[float(s) for s in scores]))
        self._timer.add_time((time.perf_counter() - start) * 1000.0, report=self.settings.verbose)
        for detection in detections:
            self._callbacks(detection)

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

    def _infer(self, batch: torch.Tensor) -> np.ndarray:
        """The person rows (N, `PERSON_ROWS`, candidates) of an RGB batch (N, 3, height, width) in [0, 1] on this
        detector's stream; only those rows come back to the CPU."""
        assert self._stream is not None
        name: str = self._engine.input_names[0]
        with torch.cuda.stream(self._stream):
            output: torch.Tensor = self._engine.run({name: batch.to(self._engine.dtype(name))})[self._engine.output_names[0]]
            rows: torch.Tensor = output[:, :PERSON_ROWS].float().cpu()      # waits for the stream
        return rows.numpy()
