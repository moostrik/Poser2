"""smile's face finder: it judges the faces the detector found (`modules.inference.face.FaceDetector`) for a portrait
and keeps the frame to cut it from.

Each `FaceFrame` the detector hands over is judged on this one's own thread: every face against the gates (score, eye
distance, the portrait crop's distance from the image edge, head pose, eyes; `quality`), the subject is the largest
qualifying face, else the largest at all, and the judged faces go to the frame callbacks as `FaceObservations`, for
the overlay and the sequencer's charge. The capture (`best()`) is the portrait cut from the last frame the subject
qualified in, else the last frame with a face, from that frame's own CPU frame (`FaceFrame.image.frame`)."""

import logging
from dataclasses import dataclass
from threading import Event, Lock, Thread

import numpy as np

from modules.camera import upright_image
from modules.inference.face import Face, FaceDetectorSettings, FaceFrame

from .portrait import crop_edge, portrait_box, portrait_crop
from .quality import FaceObservation, FaceObservations, FaceObservationsCallback, Rejection, distance, gate, readout, rejection_label
from .settings import FaceFinderSettings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _Judged:
    face: Face
    edge: float
    eyes: float | None
    rejection: Rejection | None
    limit: float
    distance: float


@dataclass(frozen=True)
class _Saved:
    """A detector's frame and the subject in it, for the capture."""
    frame: FaceFrame
    face: Face


class FaceFinder(Thread):
    """See the module docstring."""

    def __init__(self, settings: FaceFinderSettings, detector_settings: FaceDetectorSettings, composition_width: int,
                 composition_height: int) -> None:
        super().__init__(daemon=True, name='FaceFinder')
        self.settings: FaceFinderSettings = settings
        self._detector_settings: FaceDetectorSettings = detector_settings     # its pitch zero, which the calibration sets
        self._composition_width: int = composition_width
        self._composition_height: int = composition_height
        self._running: bool = False
        self._wake: Event = Event()
        self._input_lock: Lock = Lock()
        self._frame: FaceFrame | None = None        # the newest frame handed over, not judged yet
        self._saved_lock: Lock = Lock()
        self._qualified: _Saved | None = None       # the last frame in which the subject qualified
        self._last: _Saved | None = None            # the last frame with a subject at all
        self._nose_ratio: float | None = None       # of the last subject, for the pitch calibration
        self._callback_lock: Lock = Lock()
        self._callbacks: list[FaceObservationsCallback] = []
        self.settings.bind(FaceFinderSettings.calibrate, self._on_calibrate)

    # ── Lifecycle ─────────────────────────────────────────────────────

    def start(self) -> None:
        self._running = True
        super().start()

    def stop(self) -> None:
        self.settings.unbind(FaceFinderSettings.calibrate, self._on_calibrate)
        if self._running:
            self._running = False
            self._wake.set()
            self.join()

    def run(self) -> None:
        while self._running:
            self._wake.wait()
            self._wake.clear()
            with self._input_lock:
                frame, self._frame = self._frame, None
            if frame is None:
                continue
            try:
                self.judge(frame)
            except Exception:
                # nothing may end this thread: without it nobody is ever chosen. The frame is skipped
                logger.exception('Judging the faces failed')

    # ── Input: stores and returns ─────────────────────────────────────

    def submit(self, frame: FaceFrame) -> None:
        """A detector's frame (its face callback, the detector's thread): kept as the newest, judged on this thread."""
        with self._input_lock:
            self._frame = frame
        self._wake.set()

    # ── Judging ───────────────────────────────────────────────────────

    def judge(self, frame: FaceFrame) -> FaceObservations:
        """The faces of a detector's frame judged, the subject kept for the capture, and handed to the callbacks."""
        s: FaceFinderSettings = self.settings
        width, height = frame.width, frame.height
        judged: list[_Judged] = []
        for face in frame.faces:
            edge: float = crop_edge(face, s.zoom, s.offset, self._composition_width, self._composition_height, width, height)
            rejection: Rejection | None = gate(face.score, face.eye_dist, edge, face.pose, s.confidence, s.eye_min, s.crop_margin,
                                               s.max_yaw, s.max_pitch, s.max_roll)
            eyes: float | None = None
            if rejection is None:
                # the eyes count only for faces that passed the checks before them
                eyes = face.eyes
                if eyes < s.eye_open:
                    rejection = Rejection.EYES
            # the setting the failed check compared to, for the label
            limits: dict[Rejection, float] = {Rejection.LOW_SCORE: s.confidence, Rejection.SMALL: s.eye_min, Rejection.OUTSIDE: s.crop_margin,
                                              Rejection.TURNED: s.max_yaw, Rejection.PITCHED: s.max_pitch,
                                              Rejection.TILTED: s.max_roll, Rejection.EYES: s.eye_open}
            limit: float = limits[rejection] if rejection is not None else 0.0
            far: float = distance(face.score, face.eye_dist, edge, face.pose, eyes, s.confidence, s.eye_min, s.crop_margin,
                                  s.max_yaw, s.max_pitch, s.max_roll, s.eye_open, s.falloff)
            judged.append(_Judged(face, edge, eyes, rejection, limit, far))

        # the subject: the largest qualifying face, or the largest face at all
        accepted: list[_Judged] = [j for j in judged if j.rejection is None]
        candidates: list[_Judged] = accepted if accepted else judged
        subject: _Judged | None = max(candidates, key=lambda j: j.face.width * j.face.height) if candidates else None
        if subject is not None:
            self._nose_ratio = subject.face.pose.nose_ratio
            with self._saved_lock:
                self._last = _Saved(frame, subject.face)
                if subject.rejection is None:
                    self._qualified = self._last

        observations = FaceObservations(tuple(self._observation(j, j is subject, width, height) for j in judged), frame.timestamp)
        s.result = self._result(observations.subject)
        with self._callback_lock:
            callbacks: list[FaceObservationsCallback] = list(self._callbacks)
        for callback in callbacks:
            try:
                callback(observations)
            except Exception:
                logger.exception('Error in a face callback')
        return observations

    def _observation(self, j: _Judged, subject: bool, width: int, height: int) -> FaceObservation:
        s: FaceFinderSettings = self.settings
        x0, y0, x1, y1 = j.face.bbox
        crop: tuple[float, float, float, float] = portrait_box(j.face, s.zoom, s.offset, self._composition_width, self._composition_height)
        return FaceObservation(
            bbox=(float(x0) / width, float(y0) / height, float(x1) / width, float(y1) / height),
            crop=(crop[0] / width, crop[1] / height, crop[2] / width, crop[3] / height),
            kps=tuple((float(k[0]) / width, float(k[1]) / height) for k in j.face.kps),
            score=round(j.face.score, 3),
            eye_dist=round(j.face.eye_dist, 3),
            edge=round(j.edge, 3),
            pose=j.face.pose,
            eyes=j.eyes,
            rejection=j.rejection,
            limit=j.limit,
            distance=round(j.distance, 3),
            subject=subject,
        )

    @staticmethod
    def _result(subject: FaceObservation | None) -> str:
        if subject is None:
            return 'no face'
        if subject.rejection is None:
            return readout(subject)
        return f'{readout(subject)} | {rejection_label(subject)}'

    def _on_calibrate(self, _=None) -> None:
        """The face in front of the camera looks straight at it: its nose position is the detector's pitch zero."""
        ratio: float | None = self._nose_ratio
        if ratio is None:
            return
        self._detector_settings.pitch_zero = round(min(max(ratio, 0.3), 0.7), 2)

    # ── The capture ───────────────────────────────────────────────────

    def best(self) -> np.ndarray | None:
        """The portrait crop of the last frame in which the subject qualified, else of the last frame
        with a face, cut from that frame's own CPU frame, turned and flipped as the faces were found; None when no
        face was seen since the reset."""
        with self._saved_lock:
            saved: _Saved | None = self._qualified if self._qualified is not None else self._last
        if saved is None or saved.frame.image.frame is None:
            return None
        s: FaceFinderSettings = self.settings
        return portrait_crop(upright_image(saved.frame.image.frame), saved.face, s.zoom, s.offset, self._composition_width,
                             self._composition_height)

    def reset(self) -> None:
        with self._saved_lock:
            self._qualified = None
            self._last = None

    # ── Callbacks ─────────────────────────────────────────────────────

    def add_frame_callback(self, callback: FaceObservationsCallback) -> None:
        with self._callback_lock:
            if callback not in self._callbacks:
                self._callbacks.append(callback)

    def remove_frame_callback(self, callback: FaceObservationsCallback) -> None:
        with self._callback_lock:
            if callback in self._callbacks:
                self._callbacks.remove(callback)

    def clear_frame_callbacks(self) -> None:
        with self._callback_lock:
            self._callbacks = []
