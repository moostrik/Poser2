import cv2
import logging
import numpy as np
import onnxruntime as ort

from modules.face.FaceDetector import Face

logger = logging.getLogger(__name__)

# 106 point landmark model (insightface antelopev2 '2d106det'), for the eye contours.
# Preprocessing follows insightface/model_zoo/landmark.py: a square crop of 1.5 times the face box,
# and the normalisation is in the model graph. Here the crop is also rotated to level the eyes.

INPUT_SIZE: int = 192
CROP_SCALE: float = 1.5


class FaceLandmarks():
    def __init__(self, model_path: str) -> None:
        if hasattr(ort, 'preload_dlls'):
            ort.preload_dlls()
        providers: list[str] = [p for p in ['CUDAExecutionProvider', 'CPUExecutionProvider'] if p in ort.get_available_providers()]
        self.session: ort.InferenceSession = ort.InferenceSession(model_path, providers=providers)
        self.input_name: str = self.session.get_inputs()[0].name
        self.output_name: str = self.session.get_outputs()[0].name
        logger.info('FaceLandmarks running on %s', self.session.get_providers()[0])

    def detect(self, image: np.ndarray, face: Face, roll: float = 0.0) -> np.ndarray:
        """The 106 landmarks of the face, in image pixels; `roll` in degrees levels the crop."""
        x0, y0, x1, y1 = face.bbox
        centre: tuple[float, float] = (float(x0 + x1) * 0.5, float(y0 + y1) * 0.5)
        scale: float = INPUT_SIZE / (max(face.width, face.height) * CROP_SCALE)
        # rotate and scale about the face centre, then move the centre to the middle of the input
        matrix: np.ndarray = cv2.getRotationMatrix2D(centre, roll, scale)
        matrix[0, 2] += INPUT_SIZE * 0.5 - centre[0]
        matrix[1, 2] += INPUT_SIZE * 0.5 - centre[1]
        crop: np.ndarray = cv2.warpAffine(image, matrix, (INPUT_SIZE, INPUT_SIZE), borderValue=(0, 0, 0))

        blob: np.ndarray = cv2.dnn.blobFromImage(crop, 1.0, (INPUT_SIZE, INPUT_SIZE), (0.0, 0.0, 0.0), swapRB=True)
        pred: np.ndarray = self.session.run([self.output_name], {self.input_name: blob})[0][0]
        points: np.ndarray = (pred.reshape((-1, 2)) + 1.0) * (INPUT_SIZE // 2)

        inverse: np.ndarray = cv2.invertAffineTransform(matrix)
        return cv2.transform(points[None, :, :].astype(np.float64), inverse)[0]
