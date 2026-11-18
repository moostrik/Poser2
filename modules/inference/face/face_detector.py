"""The face detector: the faces of uploaded images, through SCRFD and the 106-point landmarks, on TensorRT or ONNX
Runtime.

It takes the uploaded images (`GpuImage`), so nothing of an image is worked on the CPU: it is fitted into SCRFD's
640 input on the GPU (scaled to fit, top left, as insightface letterboxes it), and each face's square crop of 1.5
times its box, levelled by its roll, is cut for the landmarks on the GPU too, all of an image's faces as one batch.
Out goes a `FaceFrame` per image: its faces with what their points measure (`measures`), and the image they were
found in, carried unread. A newer image of a camera replaces one still waiting: the latest wins."""

import logging
import time
from threading import Lock

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from modules.utils import PerformanceTimer

from ..engine import Engine, make_engine
from ..model_worker import ModelWorker
from ..source import GpuImage
from .decode import INPUT_SIZE, decode
from .io import Face, FaceFrame, FaceFrameCallback
from .measures import eye_distance, eye_openness, head_pose
from .settings import DETECTION_MODEL, LANDMARKS_MODEL, FaceDetectorSettings

logger = logging.getLogger(__name__)

INPUT_MEAN: float = 127.5
INPUT_STD: float = 128.0
LANDMARKS_SIZE: int = 192
CROP_SCALE: float = 1.5


def landmark_matrix(face_box: np.ndarray, roll: float) -> np.ndarray:
    """The affine (2 x 3) from the image to a face's landmark input: the square of `CROP_SCALE` times the box, about
    its center, rotated by `roll` degrees to level the eyes, `LANDMARKS_SIZE` wide."""
    x0, y0, x1, y1 = face_box
    center: tuple[float, float] = (float(x0 + x1) * 0.5, float(y0 + y1) * 0.5)
    scale: float = LANDMARKS_SIZE / (max(float(x1 - x0), float(y1 - y0)) * CROP_SCALE)
    matrix: np.ndarray = cv2.getRotationMatrix2D(center, roll, scale)
    matrix[0, 2] += LANDMARKS_SIZE * 0.5 - center[0]
    matrix[1, 2] += LANDMARKS_SIZE * 0.5 - center[1]
    return matrix


class FaceDetector(ModelWorker[GpuImage, FaceDetectorSettings]):
    def __init__(self, settings: FaceDetectorSettings) -> None:
        super().__init__(settings, "FaceDetector")
        self._detection: Engine = make_engine(settings.model_type, settings.model_file(DETECTION_MODEL))
        self._landmarks: Engine = make_engine(settings.model_type, settings.model_file(LANDMARKS_MODEL))
        self._callback_lock: Lock = Lock()
        self._callbacks: list[FaceFrameCallback] = []   # run in registration order, on this thread
        self._timer: PerformanceTimer = PerformanceTimer(name="Face Detection", sample_count=200, report_interval=100, color='magenta', omit_init=10)

    def add_face_callback(self, callback: FaceFrameCallback) -> None:
        with self._callback_lock:
            if callback not in self._callbacks:
                self._callbacks.append(callback)

    def remove_face_callback(self, callback: FaceFrameCallback) -> None:
        with self._callback_lock:
            if callback in self._callbacks:
                self._callbacks.remove(callback)

    def submit(self, image: GpuImage) -> None:
        """An uploaded image to look at; the newest per camera is kept."""
        self._put(image.cam_id, image)

    def _process(self, items: dict[int, GpuImage]) -> None:
        for image in items.values():
            try:
                frame: FaceFrame = self.detect(image)
            except Exception:
                logger.exception("Face detector error")
                continue
            with self._callback_lock:
                callbacks: list[FaceFrameCallback] = list(self._callbacks)
            for callback in callbacks:
                try:
                    callback(frame)
                except Exception:
                    logger.exception("Error in a face callback")

    def _setup(self) -> None:
        assert self._stream is not None
        self._detection.setup(self._stream)
        self._landmarks.setup(self._stream)
        with torch.cuda.stream(self._stream):
            warm_up: torch.Tensor = torch.zeros((1, 3, INPUT_SIZE, INPUT_SIZE), dtype=torch.float32, device='cuda')
            warm_up_landmarks: torch.Tensor = torch.zeros((1, 3, LANDMARKS_SIZE, LANDMARKS_SIZE), dtype=torch.float32, device='cuda')
        self._infer(self._detection, warm_up)
        self._infer(self._landmarks, warm_up_landmarks)
        logger.info("Face detector: %s and %s ready (%s)", DETECTION_MODEL, LANDMARKS_MODEL, self.settings.model_type.name)

    def detect(self, image: GpuImage) -> FaceFrame:
        """The faces of one uploaded image, on this thread (or a caller's, before `start`, for a test)."""
        if self._stream is None:
            self._prepare()
        assert self._stream is not None
        start: float = time.perf_counter()
        height, width = int(image.image.shape[1]), int(image.image.shape[2])
        with torch.cuda.stream(self._stream):
            self._stream.wait_event(image.ready)
            pixels: torch.Tensor = image.image.unsqueeze(0).float() * 255.0      # RGB, 0 to 255, as the models take it
            scale: float = INPUT_SIZE / max(height, width)
            fitted: torch.Tensor = F.interpolate(pixels, size=(int(height * scale), int(width * scale)), mode='bilinear', align_corners=False)
            detection_input: torch.Tensor = torch.zeros((1, 3, INPUT_SIZE, INPUT_SIZE), dtype=torch.float32, device='cuda')
            detection_input[:, :, :fitted.shape[2], :fitted.shape[3]] = fitted
            detection_input = (detection_input - INPUT_MEAN) / INPUT_STD
        found = decode(self._infer(self._detection, detection_input), scale, self.settings.threshold)

        faces: list[Face] = []
        if found:
            poses = [head_pose(kps, self.settings.pitch_zero) for _, _, kps in found]
            matrices: list[np.ndarray] = [landmark_matrix(box, pose.roll) for (box, _, _), pose in zip(found, poses)]
            landmarks: list[np.ndarray] = self._landmark_points(pixels, matrices, width, height)
            for (box, score, kps), pose, points in zip(found, poses, landmarks):
                faces.append(Face(bbox=box, score=score, kps=kps, landmarks=points, pose=pose,
                                  eyes=eye_openness(points, pose.roll), eye_dist=eye_distance(kps) / float(width)))
        self._timer.add_time((time.perf_counter() - start) * 1000.0, report=self.settings.verbose)
        return FaceFrame(cam_id=image.cam_id, timestamp=image.timestamp, width=width, height=height, faces=tuple(faces), image=image)

    def _landmark_points(self, pixels: torch.Tensor, matrices: list[np.ndarray], width: int, height: int) -> list[np.ndarray]:
        """The 106 landmarks of each face, in the image's pixels: its crop cut on the GPU through the inverse of its
        matrix (a pixel center in the crop looks up where it came from, as `cv2.warpAffine` does), all faces as one
        batch."""
        assert self._stream is not None
        size: int = LANDMARKS_SIZE
        inverses: list[np.ndarray] = [cv2.invertAffineTransform(matrix) for matrix in matrices]
        with torch.cuda.stream(self._stream):
            ys, xs = torch.meshgrid(torch.arange(size, device='cuda', dtype=torch.float32),
                                    torch.arange(size, device='cuda', dtype=torch.float32), indexing='ij')
            target: torch.Tensor = torch.stack((xs, ys, torch.ones_like(xs)), dim=-1).reshape(-1, 3)        # crop pixels
            inverse: torch.Tensor = torch.from_numpy(np.stack(inverses)).to(device='cuda', dtype=torch.float32)     # N x 2 x 3
            source: torch.Tensor = torch.einsum('nij,pj->npi', inverse, target)                              # image pixels
            grid: torch.Tensor = torch.stack(((2.0 * source[..., 0] + 1.0) / width - 1.0,
                                              (2.0 * source[..., 1] + 1.0) / height - 1.0), dim=-1).reshape(-1, size, size, 2)
            crops: torch.Tensor = F.grid_sample(pixels.expand(len(matrices), -1, -1, -1), grid, mode='bilinear',
                                                padding_mode='zeros', align_corners=False)
        predictions: np.ndarray = self._infer(self._landmarks, crops)[0]
        points: list[np.ndarray] = []
        for prediction, inverse_matrix in zip(predictions, inverses):
            crop_points: np.ndarray = (prediction.reshape((-1, 2)) + 1.0) * (size // 2)
            points.append(cv2.transform(crop_points[None, :, :].astype(np.float64), inverse_matrix)[0])
        return points

    def _infer(self, engine: Engine, batch: torch.Tensor) -> list[np.ndarray]:
        """A model's outputs, in its order, back on the CPU, for a float32 batch on this detector's stream."""
        assert self._stream is not None
        name: str = engine.input_names[0]
        with torch.cuda.stream(self._stream):
            outputs: dict[str, torch.Tensor] = engine.run({name: batch.to(engine.dtype(name))})
            on_cpu: list[torch.Tensor] = [outputs[output].float().cpu() for output in engine.output_names]    # waits for the stream
        return [output.numpy() for output in on_cpu]
