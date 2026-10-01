"""The app's image source: the frames of a camera device (Vimba or webcam), or, with `use_image` on,
an image from `input/`, whether a device was found or not. A missing device is a warning and the
`device` readout says `none`; there is no automatic fallback to the image, so a failed camera on
site stays visible."""

import logging
import os
from threading import Lock, Thread
from time import sleep
from typing import Any, Callable, Protocol

import cv2
import numpy as np

from modules.cam.settings import CameraSettings

logger = logging.getLogger(__name__)

IMAGE_FOLDER: str = 'data/input'
IMAGE_EXTENSIONS: tuple[str, ...] = ('.jpg', '.jpeg', '.png')
IMAGE_FPS: float = 25.0

FrameCallback = Callable[[np.ndarray], None]


class Device(Protocol):
    """A camera device: VimbaCam or WebCam. Frames come through the colour callback."""
    def open(self) -> bool: ...
    def close(self) -> None: ...
    def startCapture(self) -> None: ...
    def setColorCallback(self, callback: FrameCallback) -> None: ...
    def clearColorCallbacks(self) -> None: ...
    def setRotate90(self, rotate90: bool) -> None: ...


def cover(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize the image to cover width x height and crop the middle: no distortion."""
    image_height, image_width = image.shape[:2]
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
    def __init__(self, settings: CameraSettings, size: tuple[int, int], device: Device | None, device_name: str = 'none', folder: str = IMAGE_FOLDER) -> None:
        self.settings: CameraSettings = settings
        self._width, self._height = size
        self._device: Device | None = device
        self._device_name: str = device_name
        self._folder: str = folder

        self._mutex: Lock = Lock()
        self._callbacks: list[FrameCallback] = []
        self._frame: np.ndarray = np.zeros((self._height, self._width, 3), dtype=np.uint8)
        self._image: np.ndarray | None = None

        self._running: bool = False
        self._thread: Thread = Thread(target=self._run, daemon=True)

        self.settings.bind(CameraSettings.image, self._on_image)
        self.settings.bind(CameraSettings.reload, self._on_reload)

    # LIFECYCLE
    def open(self) -> bool:
        self.scan()
        self._load()
        if self._device is not None:
            try:
                opened: bool = self._device.open()
            except Exception as e:
                logger.warning('Camera %s failed to open: %s', self._device_name, e)
                opened = False
            if opened:
                self._device.setColorCallback(self._on_device_frame)
            else:
                logger.warning('No camera device (%s): only the image source is available', self._device_name)
                self._device = None
        self.settings.device = self._device_name if self._device is not None else 'none'
        self._running = True
        self._thread.start()
        return True

    def startCapture(self) -> None:
        if self._device is not None:
            self._device.startCapture()

    def close(self) -> None:
        self.settings.unbind(CameraSettings.image, self._on_image)
        self.settings.unbind(CameraSettings.reload, self._on_reload)
        if self._running:
            self._running = False
            self._thread.join()
        if self._device is not None:
            self._device.clearColorCallbacks()
            self._device.close()

    def setRotate90(self, rotate90: bool) -> None:
        # the image is fitted to the output size as it is
        if self._device is not None:
            self._device.setRotate90(rotate90)

    # FRAMES
    def getColorImg(self) -> np.ndarray:
        with self._mutex:
            return self._frame

    def setColorCallback(self, callback: FrameCallback) -> None:
        with self._mutex:
            self._callbacks.append(callback)

    def clearColorCallbacks(self) -> None:
        with self._mutex:
            self._callbacks = []

    def _on_device_frame(self, frame: np.ndarray) -> None:
        if self.settings.use_image:
            return
        self._emit(frame)

    def _run(self) -> None:
        # the image source: emitted at its own rate while it is in use
        while self._running:
            image: np.ndarray | None = self._image
            if self.settings.use_image and image is not None:
                self._emit(image)
                sleep(1.0 / IMAGE_FPS)
            else:
                sleep(0.01)

    def _emit(self, frame: np.ndarray) -> None:
        with self._mutex:
            self._frame = frame
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
