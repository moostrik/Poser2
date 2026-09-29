import cv2
import numpy as np
import time
from threading import Thread, Lock
from typing import Callable

from modules.face.FaceDetector import FaceDetector, Face
from modules.face.settings import FaceSettings

class FaceMessage():
    def __init__(self, image: np.ndarray, size: float, score: float, stable: bool) -> None:
        self.image: np.ndarray = image      # rotated portrait crop at the generator's portrait size
        self.size: float = round(size, 3)   # face height relative to image height
        self.score: float = round(score, 3)
        self.stable: bool = stable          # found in enough consecutive frames
        self.time: float = time.time()

MessageCallback = Callable[[FaceMessage], None]

def largest_face(faces: list[Face], min_score: float) -> Face | None:
    faces = [f for f in faces if f.score >= min_score]
    if not faces:
        return None
    return max(faces, key=lambda f: f.width * f.height)

def portrait_crop(img: np.ndarray, face: Face, min_size: float, portrait_zoom: float, portrait_offset: float, portrait_width: int, portrait_height: int) -> tuple[np.ndarray, float]:
    """Crop a portrait around the face, rotated so the eyes are level (as ComfyUI's face crop did).

    zoom: portrait width relative to the face height; offset: shift up, relative to the face height.
    """
    image_height, image_width = img.shape[:2]
    x0, y0, x1, y1 = face.bbox
    real_size: float = y1 - y0
    normalized_size: float = real_size / float(image_height)

    if normalized_size < min_size:
        raise ValueError('face to small')

    # eye line -> roll angle; order the eyes left to right in the image
    eye_a, eye_b = face.kps[0], face.kps[1]
    if eye_a[0] > eye_b[0]:
        eye_a, eye_b = eye_b, eye_a
    angle: float = float(np.arctan2(eye_b[1] - eye_a[1], eye_b[0] - eye_a[0]))
    axis_x: np.ndarray = np.array([np.cos(angle), np.sin(angle)])     # portrait x axis in the image
    axis_y: np.ndarray = np.array([-np.sin(angle), np.cos(angle)])    # portrait y axis (down) in the image

    crop_width: float = real_size * portrait_zoom
    crop_height: float = crop_width * portrait_height / portrait_width
    centre: np.ndarray = np.array([(x0 + x1) * 0.5, (y0 + y1) * 0.5]) - axis_y * portrait_offset * real_size

    half_x: np.ndarray = axis_x * crop_width * 0.5
    half_y: np.ndarray = axis_y * crop_height * 0.5
    corners: np.ndarray = np.array([centre - half_x - half_y, centre + half_x - half_y, centre + half_x + half_y, centre - half_x + half_y])
    if corners[:, 0].min() < 0 or corners[:, 0].max() > image_width or corners[:, 1].min() < 0 or corners[:, 1].max() > image_height:
        raise ValueError('portrait out of bounds')

    # map each portrait pixel to the image: origin at the top left corner, axes scaled to pixels
    scale: float = crop_width / portrait_width
    origin: np.ndarray = corners[0]
    matrix: np.ndarray = np.array([[axis_x[0] * scale, axis_y[0] * scale, origin[0]],
                                   [axis_x[1] * scale, axis_y[1] * scale, origin[1]]])
    portrait_img: np.ndarray = cv2.warpAffine(img, matrix, (portrait_width, portrait_height), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
    return portrait_img, normalized_size

class FaceFinder(Thread):
    def __init__(self, settings: FaceSettings, composition_width: int, composition_height: int, model_path: str = 'files/scrfd_10g_bnkps.onnx') -> None:
        super().__init__()
        self.settings: FaceSettings = settings
        self._input_mutex: Lock = Lock()
        self._thread_running: bool = False
        self._image: np.ndarray | None = None
        self._message_callbacks: list[MessageCallback] = []

        self._composition_width: int = composition_width
        self._composition_height: int = composition_height
        self._model_path: str = model_path

        self._consecutive: int = 0

    def stop(self) -> None:
        if (self._thread_running):
            self._thread_running = False
            self.join()

    def run(self) -> None:
        self._thread_running = True
        detector: FaceDetector = FaceDetector(self._model_path)
        while self._thread_running:
            image: np.ndarray | None = self.get_image()
            self.set_image(None)
            if image is not None:
                s: FaceSettings = self.settings
                face: Face | None = largest_face(detector.detect(image, 0.5), s.confidence)
                try:
                    if face is None:
                        raise ValueError('no face found')
                    portrait_img, size = portrait_crop(image, face, s.size_filter, s.zoom, s.offset, self._composition_width, self._composition_height)
                    self._consecutive += 1
                    self.message_callback(FaceMessage(portrait_img, size, face.score, self._consecutive >= s.stable_frames))
                except ValueError as e:
                    self._consecutive = 0
                    self.error_callback(e)
            else:
                time.sleep(0.01)

    def reset(self) -> None:
        self._consecutive = 0

    def set_image(self, image: np.ndarray | None) -> None:
        with self._input_mutex:
            self._image = image
    def get_image(self) -> np.ndarray | None:
        with self._input_mutex:
            return self._image

    def message_callback(self, message: FaceMessage) -> None:
        self.settings.result = 'size: ' + str(message.size) + ', score: ' + str(message.score) + ', stable: ' + str(message.stable)
        with self._input_mutex:
            for c in self._message_callbacks: c(message)

    def error_callback(self, error: ValueError) -> None:
        self.settings.result = 'error: ' + str(error)

    def add_message_callback(self, callback: MessageCallback) -> None:
        with self._input_mutex:
            self._message_callbacks.append(callback)

    def remove_message_callback(self, callback: MessageCallback) -> None:
        with self._input_mutex:
            if callback in self._message_callbacks:
                self._message_callbacks.remove(callback)

    def clear_message_callbacks(self) -> None:
        with self._input_mutex:
            self._message_callbacks = []
