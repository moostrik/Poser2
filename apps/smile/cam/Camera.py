"""The app's image source: the frames of a camera device (a `CameraSource`: the Vimba camera or the
webcam, `modules/camera/`), or, with `use_image` on, an image from `input/`, whether a device was
found or not; or, during an online turn, the frames of a phone (`set_online`, `push_online`: SMILE
Online, modules/online/). Every frame is fitted to the output size here (`cover`: scaled to fill it,
the middle kept, no distortion); the device only rotates and flips. A missing device is a warning
and the `device` readout says `none`; there is no automatic fallback to the image, so a failed
camera on site stays visible. The device is kept when it did not connect: one that keeps trying
(Vimba) delivers when it is there, and the readout follows it. Every frame has the time it arrived
(`frame()`, on `time.perf_counter()`, the one clock of the frames and the sequence: a device frame the time
its backend stamped), so a source that stopped is told from one that delivers (`stalled`)."""

import logging
import os
from threading import Lock, Thread
from time import perf_counter, sleep
from typing import Any, Callable

import cv2
import numpy as np

from apps.smile.cam.settings import CameraSettings
from modules.camera import CameraFrame, CameraSource

logger = logging.getLogger(__name__)

IMAGE_FOLDER: str = 'apps/smile/data/input'
IMAGE_EXTENSIONS: tuple[str, ...] = ('.jpg', '.jpeg', '.png')
IMAGE_FPS: float = 25.0
STALLED_AFTER: float = 3.0      # s without a frame: the source has stopped

FrameCallback = Callable[[np.ndarray], None]


def cover(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize the image to cover width x height and crop the middle: no distortion. An image of that
    size is returned as it is."""
    image_height, image_width = image.shape[:2]
    if (image_width, image_height) == (width, height):
        return image
    scale: float = max(width / image_width, height / image_height)
    resized: np.ndarray = cv2.resize(image, (max(width, round(image_width * scale)), max(height, round(image_height * scale))), interpolation=cv2.INTER_AREA)
    x0: int = (resized.shape[1] - width) // 2
    y0: int = (resized.shape[0] - height) // 2
    return np.ascontiguousarray(resized[y0:y0 + height, x0:x0 + width])


def list_images(folder: str) -> list[str]:
    if not os.path.isdir(folder):
        return []
    return sorted(f for f in os.listdir(folder) if f.lower().endswith(IMAGE_EXTENSIONS))


class Camera:
    def __init__(self, settings: CameraSettings, size: tuple[int, int], device: CameraSource | None, device_name: str = 'none', folder: str = IMAGE_FOLDER) -> None:
        self.settings: CameraSettings = settings
        self._width, self._height = size
        self._device: CameraSource | None = device
        self._device_name: str = device_name
        self._device_shown: str | None = None       # what the device readout says
        self._folder: str = folder

        self._mutex: Lock = Lock()
        self._callbacks: list[FrameCallback] = []
        self._frame: np.ndarray = np.zeros((self._height, self._width, 3), dtype=np.uint8)
        self._frame_time: float = 0.0       # when the newest frame arrived; 0 before the first
        self._image: np.ndarray | None = None
        self._online: bool = False          # the online source has the image: the device and the image source are muted

        self._running: bool = False
        self._thread: Thread = Thread(target=self._run, daemon=True)

        self.settings.bind(CameraSettings.image, self._on_image)
        self.settings.bind(CameraSettings.reload, self._on_reload)

    # LIFECYCLE
    def start(self) -> None:
        self.scan()
        self._load()
        if self._device is not None:
            self._device.add_frame_callback(self._on_device_frame)
            try:
                self._device.start()
            except Exception as e:
                logger.warning('Camera %s failed to start: %s', self._device_name, e)
            # also when it did not connect: a device that keeps trying (Vimba) delivers when it is there
            if not self._device.connected:
                logger.warning('No camera device (%s): only the image source is available', self._device_name)
        self._show_device()
        self._running = True
        self._thread.start()

    def stop(self) -> None:
        self.settings.unbind(CameraSettings.image, self._on_image)
        self.settings.unbind(CameraSettings.reload, self._on_reload)
        if self._running:
            self._running = False
            self._thread.join()
        if self._device is not None:
            self._device.stop()
            self._device.join()
        self.clear_frame_callbacks()

    # FRAMES
    def getColorImg(self) -> np.ndarray:
        with self._mutex:
            return self._frame

    def frame(self) -> tuple[np.ndarray, float]:
        """The newest frame and when it arrived: a source that stopped keeps its last frame, with its old time."""
        with self._mutex:
            return self._frame, self._frame_time

    @property
    def stalled(self) -> bool:
        """No frame for a while (or none yet): no device, a camera that stopped, or no image to show."""
        with self._mutex:
            return perf_counter() - self._frame_time > STALLED_AFTER

    def add_frame_callback(self, callback: FrameCallback) -> None:
        with self._mutex:
            self._callbacks.append(callback)

    def clear_frame_callbacks(self) -> None:
        with self._mutex:
            self._callbacks = []

    def _on_device_frame(self, frame: CameraFrame) -> None:
        # on the device's thread
        if self.settings.use_image or self._online:
            return
        # the one fit of every source to the output size: the device sends its frames as they are
        self._emit(cover(frame.image, self._width, self._height), frame.time)

    # ONLINE SOURCE: the frames of a phone during its turn (modules/online/), in place of the device
    # and the image; the device keeps running underneath and is back the moment the turn ends
    @property
    def online(self) -> bool:
        return self._online

    def set_online(self, active: bool) -> None:
        with self._mutex:
            self._online = active

    def push_online(self, frame: np.ndarray) -> None:
        """A frame of the online source: fitted like any other; dropped while the source is off."""
        if not self._online:
            return
        self._emit(cover(frame, self._width, self._height), perf_counter())

    def _run(self) -> None:
        # the image source: emitted at its own rate while it is in use
        while self._running:
            self._show_device()
            image: np.ndarray | None = self._image
            if self.settings.use_image and image is not None and not self._online:
                self._emit(image, perf_counter())
                sleep(1.0 / IMAGE_FPS)
            else:
                sleep(0.01)

    def _show_device(self) -> None:
        """The device readout: its name while it is connected, `none` while it is not (yet, or any more)."""
        name: str = self._device_name if self._device is not None and self._device.connected else 'none'
        if name != self._device_shown:
            self._device_shown = name
            self.settings.device = name

    def _emit(self, frame: np.ndarray, arrived: float) -> None:
        with self._mutex:
            self._frame = frame
            self._frame_time = arrived
            callbacks: list[FrameCallback] = list(self._callbacks)
        for c in callbacks:
            c(frame)

    # IMAGE SOURCE
    def scan(self) -> list[str]:
        """List the images of the folder into the panel; keep the selection if its file is still there."""
        files: list[str] = list_images(self._folder)
        self.settings.image_options = files if files else ['']
        self.settings.image_count = len(files)
        if files and self.settings.image not in files:
            self.settings.image = files[0]
        if not files:
            logger.warning('No images in %s', self._folder)
        return files

    def _load(self) -> None:
        name: str = self.settings.image
        if not name:
            self._image = None
            return
        image: np.ndarray | None = cv2.imread(os.path.join(self._folder, name), cv2.IMREAD_COLOR)
        if image is None:
            logger.warning('Failed to load image %s', name)
            self._image = None
            return
        self._image = cover(image, self._width, self._height)

    def next_image(self) -> None:
        """After a capture: move the image source to the next file of the folder, when asked to."""
        s: CameraSettings = self.settings
        if not (s.use_image and s.image_next):
            return
        files: list[str] = [f for f in s.image_options if f]
        if not files:
            return
        index: int = files.index(s.image) if s.image in files else -1
        s.image = files[(index + 1) % len(files)]

    def _on_image(self, _: Any = None) -> None:
        self._load()

    def _on_reload(self, _: Any = None) -> None:
        self.scan()
        self._load()
