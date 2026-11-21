"""The webcam backend: the PC's own camera through OpenCV (`cv2.VideoCapture`) on the camera
contract, for development without the installation's camera. `start()` opens the device; one that
cannot be opened is a warning and the camera stays unconnected: a webcam is not searched for again."""

import logging
import time
from threading import Thread

import cv2
import numpy as np

from .. import CameraFrame, CameraReadings, CameraSettings, FrameType, CameraFrameCallback, TrackletCallback
from modules.utils import Broadcast, FPS, Waker

logger = logging.getLogger(__name__)

READ_RETRY: float = 0.01        # s after a failed read before the next


class Camera(Thread):
    """A `CameraSource` on a webcam. Every frame carries the settings' `rotate90` and flips as flags. No detector: no
    tracklets."""

    def __init__(self, settings: CameraSettings, cam_id: int = 0, index: int = 0) -> None:
        super().__init__(name='WebCamera', daemon=True)
        self.settings: CameraSettings = settings
        self.cam_id: int = cam_id
        self.rotate90: bool = settings.rotate90     # how the camera is mounted, fixed at start
        self.index: int = index                 # the device's index for OpenCV

        self.frame_callbacks: Broadcast[[CameraFrame]] = Broadcast(name='webcam frame')
        self.video_fps: FPS = FPS()

        self.running: bool = False
        self._waker: Waker = Waker()            # set by stop: ends the wait after a failed read at once
        self._capture: cv2.VideoCapture | None = None

    @property
    def readings(self) -> CameraReadings:
        return self.settings.readings

    @property
    def connected(self) -> bool:
        return self._capture is not None

    def add_frame_callback(self, callback: CameraFrameCallback) -> None:
        self.frame_callbacks.add_callback(callback)

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
        self._waker.set()

    def join(self, timeout: float | None = None) -> None:
        if self.ident is not None:      # started
            super().join(timeout)
        if not self.is_alive():
            self._waker.close()

    def run(self) -> None:
        capture: cv2.VideoCapture | None = self._capture
        assert capture is not None
        try:
            while self.running:
                ok, image = capture.read()
                if ok:
                    self._emit(image)
                else:
                    self._waker.wait(time.perf_counter() + READ_RETRY)
        finally:
            # the thread first: it was reading from the device
            self._capture = None
            capture.release()

    def _emit(self, image: np.ndarray) -> None:
        """The frame becomes the contract's: as delivered, its turn and flips as flags, with its host time on the one
        clock."""
        self.video_fps.processed()
        self.settings.readings.video_fps = self.video_fps.get_rate_average()
        self.settings.readings.video_min_fps = self.video_fps.get_rate_minimum()
        self.frame_callbacks(CameraFrame(cam_id=self.cam_id, kind=FrameType.VIDEO, image=image,
                                         timestamp=time.perf_counter(), rotate90=self.rotate90,
                                         flip_v=self.settings.flip_v, flip_h=self.settings.flip_h))
