import cv2
import logging
import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)

# SCRFD face detector (insightface antelopev2 'scrfd_10g_bnkps'), the same model InstantID uses in comfy.
# Decoding follows insightface/model_zoo/scrfd.py, without the insightface package.

INPUT_SIZE: int = 640
INPUT_MEAN: float = 127.5
INPUT_STD: float = 128.0
FEAT_STRIDES: list[int] = [8, 16, 32]
NUM_ANCHORS: int = 2
NMS_THRESHOLD: float = 0.4

class Face():
    def __init__(self, bbox: np.ndarray, score: float, kps: np.ndarray) -> None:
        self.bbox: np.ndarray = bbox    # x0, y0, x1, y1 in image pixels
        self.score: float = score
        self.kps: np.ndarray = kps      # 5 x 2: eyes, nose, mouth corners

    @property
    def width(self) -> float:
        return float(self.bbox[2] - self.bbox[0])
    @property
    def height(self) -> float:
        return float(self.bbox[3] - self.bbox[1])

def distance2bbox(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    x0 = points[:, 0] - distance[:, 0]
    y0 = points[:, 1] - distance[:, 1]
    x1 = points[:, 0] + distance[:, 2]
    y1 = points[:, 1] + distance[:, 3]
    return np.stack([x0, y0, x1, y1], axis=-1)

def distance2kps(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    preds: list[np.ndarray] = []
    for i in range(0, distance.shape[1], 2):
        preds.append(points[:, i % 2] + distance[:, i])
        preds.append(points[:, i % 2 + 1] + distance[:, i + 1])
    return np.stack(preds, axis=-1).reshape((distance.shape[0], -1, 2))

class FaceDetector():
    def __init__(self, model_path: str) -> None:
        if hasattr(ort, 'preload_dlls'):
            ort.preload_dlls() # use the CUDA and cuDNN libraries from the venv (installed with torch)
        providers: list[str] = [p for p in ['CUDAExecutionProvider', 'CPUExecutionProvider'] if p in ort.get_available_providers()]
        self.session: ort.InferenceSession = ort.InferenceSession(model_path, providers=providers)
        self.input_name: str = self.session.get_inputs()[0].name
        self.output_names: list[str] = [o.name for o in self.session.get_outputs()]
        self.batched: bool = len(self.session.get_outputs()[0].shape) == 3
        self._anchor_centers: dict[int, np.ndarray] = {}
        logger.info('FaceDetector running on %s', self.session.get_providers()[0])

    def _get_anchor_centers(self, stride: int) -> np.ndarray:
        if stride not in self._anchor_centers:
            size: int = INPUT_SIZE // stride
            centers: np.ndarray = np.stack(np.mgrid[:size, :size][::-1], axis=-1).astype(np.float32)
            centers = (centers * stride).reshape((-1, 2))
            self._anchor_centers[stride] = np.stack([centers] * NUM_ANCHORS, axis=1).reshape((-1, 2))
        return self._anchor_centers[stride]

    def detect(self, image: np.ndarray, threshold: float = 0.5) -> list[Face]:
        # letterbox the BGR image into the top left of a square input
        image_height, image_width = image.shape[:2]
        scale: float = INPUT_SIZE / max(image_height, image_width)
        resized: np.ndarray = cv2.resize(image, (int(image_width * scale), int(image_height * scale)))
        det_image: np.ndarray = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
        det_image[:resized.shape[0], :resized.shape[1]] = resized

        blob: np.ndarray = cv2.dnn.blobFromImage(det_image, 1.0 / INPUT_STD, (INPUT_SIZE, INPUT_SIZE), (INPUT_MEAN, INPUT_MEAN, INPUT_MEAN), swapRB=True)
        outputs: list = self.session.run(self.output_names, {self.input_name: blob})

        scores_list: list[np.ndarray] = []
        bboxes_list: list[np.ndarray] = []
        kps_list: list[np.ndarray] = []
        fmc: int = len(FEAT_STRIDES)
        for idx, stride in enumerate(FEAT_STRIDES):
            scores: np.ndarray = outputs[idx][0] if self.batched else outputs[idx]
            bbox_preds: np.ndarray = (outputs[idx + fmc][0] if self.batched else outputs[idx + fmc]) * stride
            kps_preds: np.ndarray = (outputs[idx + fmc * 2][0] if self.batched else outputs[idx + fmc * 2]) * stride

            centers: np.ndarray = self._get_anchor_centers(stride)
            pos: np.ndarray = np.where(scores.reshape(-1) >= threshold)[0]
            scores_list.append(scores.reshape(-1)[pos])
            bboxes_list.append(distance2bbox(centers, bbox_preds)[pos])
            kps_list.append(distance2kps(centers, kps_preds)[pos])

        all_scores: np.ndarray = np.concatenate(scores_list)
        if len(all_scores) == 0:
            return []
        all_bboxes: np.ndarray = np.concatenate(bboxes_list) / scale
        all_kps: np.ndarray = np.concatenate(kps_list) / scale

        rects: list = [[float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])] for b in all_bboxes]
        keep = cv2.dnn.NMSBoxes(rects, all_scores.tolist(), threshold, NMS_THRESHOLD)
        faces: list[Face] = [Face(all_bboxes[i], float(all_scores[i]), all_kps[i]) for i in np.array(keep).reshape(-1)]
        faces.sort(key=lambda f: f.score, reverse=True)
        return faces
