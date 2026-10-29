"""The webcam backend: the PC's own camera through OpenCV (`cv2.VideoCapture`) on the camera
contract, for development without the installation's camera. `start()` opens the device; one that
cannot be opened is a warning and the camera stays unconnected: a webcam is not searched for again."""

import logging
import time
from threading import Thread

import cv2
import numpy as np

from .. import CameraFrame, CameraReadings, CameraSettings, FrameType, FrameCallback, TrackletCallback
from modules.utils import FPS

logger = logging.getLogger(__name__)


class Camera(Thread):
    """A `CameraSource` on a webcam. `rotate90` turns every frame (portrait mode, fixed at start);
    the flips come from the settings, per frame. No detector: no tracklets."""

    def __init__(self, settings: CameraSettings, rotate90: bool = False, id: int = 0, index: int = 0) -> None:
        super().__init__(name='WebCamera', daemon=True)
        self.settings: CameraSettings = settings
        self.id: int = id
        self.rotate90: bool = rotate90
        self.index: int = index                 # the device's index for OpenCV

        self.frame_callbacks: list[FrameCallback] = []
        self.video_fps: FPS = FPS(120)

        self.running: bool = False
        self._capture: cv2.VideoCapture | None = None

    @property
    def readings(self) -> CameraReadings:
        return self.settings.readings

    @property
    def connected(self) -> bool:
        return self._capture is not None

    def add_frame_callback(self, callback: FrameCallback) -> None:
        self.frame_callbacks.append(callback)

    def add_tracklet_callback(self, callback: TrackletCallback) -> None:
        pass    # no detector on this camera: nothing to deliver

    # LIFECYCLE
    def start(self) -> None:
        if self.running:
            return
        capture = cv2.VideoCapture(self.index)
        if not capture.isOpened():
            logger.warning('Webcam %d cannot be opened', self.index)
            capture.release()
            return
        self._capture = capture
        self.running = True
        super().start()

    def stop(self) -> None:
        self.running = False

    def join(self, timeout: float | None = None) -> None:
        if self.ident is not None:      # started
            super().join(timeout)

    def run(self) -> None:
        capture: cv2.VideoCapture | None = self._capture
        assert capture is not None
        try:
            while self.running:
                ok, image = capture.read()
                if ok:
                    self._emit(image)
                else:
                    time.sleep(0.01)
        finally:
            # the thread first: it was reading from the device
            self._capture = None
            capture.release()

    def _emit(self, image: np.ndarray) -> None:
        """The frame becomes the contract's: rotated and flipped, with its host time."""
        if self.rotate90:
            image = cv2.transpose(image)
        if self.settings.flip_v:
            image = cv2.flip(image, 0)
        if self.settings.flip_h:
            image = cv2.flip(image, 1)
        self.video_fps.processed()
        self.settings.readings.video_fps = self.video_fps.get_rate_average()
        self.settings.readings.video_min_fps = self.video_fps.get_rate_minimum()
        frame = CameraFrame(cam_id=self.id, kind=FrameType.VIDEO, image=image, time=time.time())
        for callback in self.frame_callbacks:
            callback(frame)
