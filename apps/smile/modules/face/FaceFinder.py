import cv2
import numpy as np
import time
from dataclasses import dataclass
from threading import Thread, Lock
from typing import Callable

from modules.face.FaceDetector import FaceDetector, Face
from modules.face.FaceLandmarks import FaceLandmarks
from modules.face.quality import FaceObservation, Pose, Rejection, centre_offset, eye_distance, eye_openness, gate, head_pose, readout, rejection_label
from modules.face.settings import FaceSettings

DETECT_THRESHOLD: float = 0.5


@dataclass(frozen=True)
class FaceFrame:
    """Every face of one camera frame, for the overlay and the sequencer's charge."""
    faces: tuple[FaceObservation, ...]
    time: float

    @property
    def subject(self) -> FaceObservation | None:
        return next((f for f in self.faces if f.subject), None)

    @property
    def qualified(self) -> bool:
        """The subject passed every check."""
        subject: FaceObservation | None = self.subject
        return subject is not None and subject.rejection is None


FrameCallback = Callable[[FaceFrame], None]


def largest(faces: list[Face]) -> Face | None:
    if not faces:
        return None
    return max(faces, key=lambda f: f.width * f.height)


def portrait_crop(img: np.ndarray, face: Face, portrait_zoom: float, portrait_offset: float, portrait_width: int, portrait_height: int) -> np.ndarray:
    """Crop a portrait around the face, rotated so the eyes are level (as ComfyUI's face crop did).

    zoom: portrait width relative to the face height; offset: shift up, relative to the face height.
    """
    x0, y0, x1, y1 = face.bbox
    real_size: float = y1 - y0

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

    # map each portrait pixel to the image: origin at the top left corner, axes scaled to pixels
    scale: float = crop_width / portrait_width
    origin: np.ndarray = centre - half_x - half_y
    matrix: np.ndarray = np.array([[axis_x[0] * scale, axis_y[0] * scale, origin[0]],
                                   [axis_x[1] * scale, axis_y[1] * scale, origin[1]]])
    # the crop may extend past the image edge: fill that part with the edge pixels instead of black
    return cv2.warpAffine(img, matrix, (portrait_width, portrait_height), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)


@dataclass(frozen=True)
class _Assessed:
    face: Face
    eye_dist: float
    centre: float
    pose: Pose
    eyes: float | None
    rejection: Rejection | None
    limit: float


@dataclass(frozen=True)
class _Saved:
    """A camera frame and the subject in it, for the capture."""
    image: np.ndarray
    face: Face


class FaceFinder(Thread):
    """Detects the faces of each camera frame and checks them (confidence, eye distance, pose, eyes). The
    subject of a frame is its largest qualifying face. The capture is the last frame in which the
    subject qualified (`best()`), or the last frame at all when none did. Every face goes to the
    frame callbacks, for the overlay."""

    def __init__(self, settings: FaceSettings, composition_width: int, composition_height: int,
                 model_path: str = 'data/models/scrfd_10g_bnkps.onnx', landmark_path: str = 'data/models/2d106det.onnx') -> None:
        super().__init__()
        self.settings: FaceSettings = settings
        self._input_mutex: Lock = Lock()
        self._saved_mutex: Lock = Lock()
        self._thread_running: bool = False
        self._image: np.ndarray | None = None
        self._frame_callbacks: list[FrameCallback] = []

        self._composition_width: int = composition_width
        self._composition_height: int = composition_height
        self._model_path: str = model_path
        self._landmark_path: str = landmark_path

        self._qualified: _Saved | None = None       # the last frame in which the subject qualified
        self._last: _Saved | None = None            # the last frame with a subject at all
        self._nose_ratio: float | None = None       # of the last subject, for the pitch calibration

        self.settings.bind(FaceSettings.calibrate, self._on_calibrate)

    def stop(self) -> None:
        self.settings.unbind(FaceSettings.calibrate, self._on_calibrate)
        if (self._thread_running):
            self._thread_running = False
            self.join()

    def _on_calibrate(self, _=None) -> None:
        """The face in front of the camera looks straight at it: its nose position is the pitch zero."""
        ratio: float | None = self._nose_ratio
        if ratio is None:
            return
        self.settings.pitch_zero = round(min(max(ratio, 0.3), 0.7), 2)

    def run(self) -> None:
        self._thread_running = True
        detector: FaceDetector = FaceDetector(self._model_path)
        landmarks: FaceLandmarks = FaceLandmarks(self._landmark_path)
        while self._thread_running:
            image: np.ndarray | None = self.get_image()
            self.set_image(None)
            if image is not None:
                self._process(image, detector, landmarks)
            else:
                time.sleep(0.01)

    def _process(self, image: np.ndarray, detector: FaceDetector, landmarks: FaceLandmarks) -> None:
        s: FaceSettings = self.settings
        now: float = time.time()
        image_height, image_width = image.shape[:2]

        assessed: list[_Assessed] = []
        for face in detector.detect(image, DETECT_THRESHOLD):
            eye_dist: float = eye_distance(face.kps) / float(image_width)
            centre: float = centre_offset(face.kps, image_width)
            pose: Pose = head_pose(face.kps, s.pitch_zero)
            rejection: Rejection | None = gate(face.score, eye_dist, centre, pose, s.confidence, s.eye_min, s.eye_max, s.max_offset,
                                               s.max_yaw, s.max_pitch, s.max_roll)
            eyes: float | None = None
            if rejection is None:
                # the landmarks only for faces that passed the cheap checks
                eyes = eye_openness(landmarks.detect(image, face, pose.roll), pose.roll)
                if eyes < s.eye_open:
                    rejection = Rejection.EYES
            # the setting the failed check compared to, for the label
            limits: dict[Rejection, float] = {Rejection.LOW_SCORE: s.confidence, Rejection.SMALL: s.eye_min, Rejection.LARGE: s.eye_max,
                                              Rejection.OFF_CENTRE: s.max_offset, Rejection.TURNED: s.max_yaw, Rejection.PITCHED: s.max_pitch,
                                              Rejection.TILTED: s.max_roll, Rejection.EYES: s.eye_open}
            limit: float = limits[rejection] if rejection is not None else 0.0
            assessed.append(_Assessed(face, eye_dist, centre, pose, eyes, rejection, limit))

        # the subject: the largest qualifying face, or the largest face at all
        subject: _Assessed | None = None
        accepted: list[_Assessed] = [a for a in assessed if a.rejection is None]
        candidates: list[_Assessed] = accepted if accepted else assessed
        if candidates:
            subject = max(candidates, key=lambda a: a.face.width * a.face.height)
        qualified: bool = subject is not None and subject.rejection is None

        if subject is not None:
            self._nose_ratio = subject.pose.nose_ratio
            with self._saved_mutex:
                self._last = _Saved(image, subject.face)
                if qualified:
                    self._qualified = self._last

        frame = FaceFrame(tuple(self._observation(a, a is subject, image_width, image_height) for a in assessed), now)
        s.result = self._result(frame.subject)
        self.frame_callback(frame)

    @staticmethod
    def _observation(a: _Assessed, subject: bool, width: int, height: int) -> FaceObservation:
        x0, y0, x1, y1 = a.face.bbox
        return FaceObservation(
            bbox=(float(x0) / width, float(y0) / height, float(x1) / width, float(y1) / height),
            kps=tuple((float(k[0]) / width, float(k[1]) / height) for k in a.face.kps),
            score=round(a.face.score, 3),
            eye_dist=round(a.eye_dist, 3),
            centre=round(a.centre, 3),
            pose=a.pose,
            eyes=a.eyes,
            rejection=a.rejection,
            limit=a.limit,
            subject=subject,
        )

    @staticmethod
    def _result(subject: FaceObservation | None) -> str:
        if subject is None:
            return 'no face'
        if subject.rejection is None:
            return readout(subject)
        return f'{readout(subject)} | {rejection_label(subject)}'

    def best(self) -> np.ndarray | None:
        """The portrait crop of the last frame in which the subject qualified, else of the last frame
        with a face; None when no face was seen since the reset."""
        with self._saved_mutex:
            saved: _Saved | None = self._qualified if self._qualified is not None else self._last
        if saved is None:
            return None
        s: FaceSettings = self.settings
        return portrait_crop(saved.image, saved.face, s.zoom, s.offset, self._composition_width, self._composition_height)

    def reset(self) -> None:
        with self._saved_mutex:
            self._qualified = None
            self._last = None

    def set_image(self, image: np.ndarray | None) -> None:
        with self._input_mutex:
            self._image = image
    def get_image(self) -> np.ndarray | None:
        with self._input_mutex:
            return self._image

    def frame_callback(self, frame: FaceFrame) -> None:
        with self._input_mutex:
            for c in self._frame_callbacks: c(frame)

    def add_frame_callback(self, callback: FrameCallback) -> None:
        with self._input_mutex:
            self._frame_callbacks.append(callback)

    def remove_frame_callback(self, callback: FrameCallback) -> None:
        with self._input_mutex:
            if callback in self._frame_callbacks:
                self._frame_callbacks.remove(callback)

    def clear_frame_callbacks(self) -> None:
        with self._input_mutex:
            self._frame_callbacks = []
