"""The app's image source: the frames of a camera device (a `CameraSource`: the Vimba camera or the
webcam, `modules/camera/`), or, with `use_image` on, an image from `input/`, whether a device was
found or not; or, during an online turn, the frames of a phone (`set_online`, `submit_online_frame`: SMILE
Online, `online/`). The device and the phone only hand their newest frame over; this one's own thread hands
the live one on whole, as a `CameraFrame`, to the callbacks, and paces the image input. A device frame keeps its
turn and flips as flags, which who shows it applies; what the screen shows of a frame is the view (`view.py`).
Every device frame also goes, as it came, to the device frame callbacks (the recorder), after the live path took it.
A missing device is a warning and the `device` readout says `none`; there is no automatic fallback to the image, so
a failed camera on site stays visible. The device is kept when it did not connect: one that keeps trying
(Vimba) delivers when it is there, and the readout follows it. Every frame has the time it arrived
(`frame()`, on `time.perf_counter()`, the one clock of the frames and the sequence: a device frame the time
its backend stamped), so a source that stopped is told from one that delivers (`stalled`)."""

import logging
import os
from threading import Lock, Thread
from time import perf_counter
from typing import Any

import cv2
import numpy as np

from ..paths import DATA
from .settings import CameraSettings
from modules.camera import CameraFrame, CameraFrameCallback, CameraSource, FrameType
from modules.utils import Broadcast, Waker

logger = logging.getLogger(__name__)

IMAGE_FOLDER: str = f'{DATA}/input'
IMAGE_EXTENSIONS: tuple[str, ...] = ('.jpg', '.jpeg', '.png')
IMAGE_FPS: float = 25.0
STALLED_AFTER: float = 3.0      # s without a frame: the source has stopped
READOUT_EVERY: float = 0.1      # s: the longest the thread waits, so the device readout follows the device
NO_FRAME: np.ndarray = np.zeros((8, 8, 3), dtype=np.uint8)     # the frame before the first: black


def list_images(folder: str) -> list[str]:
    if not os.path.isdir(folder):
        return []
    return sorted(f for f in os.listdir(folder) if f.lower().endswith(IMAGE_EXTENSIONS))


class Camera:
    def __init__(self, settings: CameraSettings, device: CameraSource | None, device_name: str = 'none', folder: str = IMAGE_FOLDER) -> None:
        self.settings: CameraSettings = settings
        self._device: CameraSource | None = device
        self._device_name: str = device_name
        self._device_shown: str | None = None       # what the device readout says
        self._folder: str = folder

        self._mutex: Lock = Lock()
        self._callbacks: Broadcast[[CameraFrame]] = Broadcast(name='camera frame')
        self._device_callbacks: Broadcast[[CameraFrame]] = Broadcast(name='device frame')
        # the newest frame, stamped when it arrived; 0 before the first
        self._frame: CameraFrame = CameraFrame(0, FrameType.VIDEO, NO_FRAME, 0.0)
        self._image: np.ndarray | None = None
        self._online: bool = False          # the online source has the image: the device and the image source are muted
        # The newest frame handed over (the device's or the phone's), for this thread to hand on
        self._lock: Lock = Lock()
        self._waker: Waker = Waker()
        self._handed: CameraFrame | None = None

        self._running: bool = False
        self._thread: Thread = Thread(target=self._run, daemon=True, name='SmileCamera')

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
            # a device that opens on its own thread (Vimba, OAK4) is not connected yet: the readout says when it is
        self._show_device()
        self._running = True
        self._thread.start()

    def stop(self) -> None:
        self.settings.unbind(CameraSettings.image, self._on_image)
        self.settings.unbind(CameraSettings.reload, self._on_reload)
        if self._running:
            with self._lock:
                self._running = False
            self._waker.set()
            self._thread.join()
        self._waker.close()
        if self._device is not None:
            self._device.stop()
            self._device.join()
        self.clear_frame_callbacks()

    # FRAMES
    def frame(self) -> CameraFrame:
        """The newest frame, stamped when it arrived: a source that stopped keeps its last frame, with its old time;
        a black one at 0 before the first."""
        with self._mutex:
            return self._frame

    @property
    def stalled(self) -> bool:
        """No frame for a while (or none yet): no device, a camera that stopped, or no image to show."""
        with self._mutex:
            return perf_counter() - self._frame.timestamp > STALLED_AFTER

    def add_frame_callback(self, callback: CameraFrameCallback) -> None:
        self._callbacks.add_callback(callback)

    def clear_frame_callbacks(self) -> None:
        self._callbacks.clear()

    def add_device_frame_callback(self, callback: CameraFrameCallback) -> None:
        """Every frame of the device as it came, on the device's thread, after the live path took it: the recorder's."""
        self._device_callbacks.add_callback(callback)

    def _on_device_frame(self, frame: CameraFrame) -> None:
        """On the device's thread: the frame handed over as it is, with the time its backend stamped, then to the
        device frame callbacks."""
        if not (self.settings.use_image or self._online):
            self._hand_over(frame)
        self._device_callbacks(frame)

    def _hand_over(self, frame: CameraFrame) -> None:
        with self._lock:
            self._handed = frame
        self._waker.set()

    # ONLINE SOURCE: the frames of a phone during its turn (`online/`), in place of the device
    # and the image; the device keeps running underneath and is back the moment the turn ends
    @property
    def online(self) -> bool:
        return self._online

    def set_online(self, active: bool) -> None:
        with self._mutex:
            self._online = active

    def submit_online_frame(self, frame: np.ndarray) -> None:
        """A frame of the online source, handed over like any other; dropped while the source is off."""
        if not self._online:
            return
        self._hand_over(CameraFrame(0, FrameType.VIDEO, frame, perf_counter()))

    def _run(self) -> None:
        """Hands each frame handed over on as it comes, and emits the image input at its own rate while it is in use;
        what a callback raises is logged and the loop goes on."""
        next_image: float = 0.0
        while True:
            self._show_device()
            image: np.ndarray | None = self._image
            image_on: bool = self.settings.use_image and image is not None and not self._online
            now: float = perf_counter()
            deadline: float = min(now + READOUT_EVERY, max(now, next_image)) if image_on else now + READOUT_EVERY
            with self._lock:
                idle: bool = self._running and self._handed is None
            if idle:
                self._waker.wait(deadline)
            with self._lock:
                if not self._running:
                    return
                handed: CameraFrame | None = self._handed
                self._handed = None
            try:
                if handed is not None and (self._online or not self.settings.use_image):
                    self._emit(handed)
                now = perf_counter()
                if image_on and image is not None and now >= next_image:
                    # on an absolute schedule: each image is due a period after the last one's due time, not after
                    # it came out
                    next_image = max(next_image + 1.0 / IMAGE_FPS, now - 1.0 / IMAGE_FPS)
                    self._emit(CameraFrame(0, FrameType.VIDEO, image, now))
            except Exception:
                logger.exception('Camera: a frame failed')

    def _show_device(self) -> None:
        """The device readout: its name while it is connected, `none` while it is not (yet, or any more)."""
        name: str = self._device_name if self._device is not None and self._device.connected else 'none'
        if name != self._device_shown:
            if name != 'none':
                logger.info('Camera device %s connected', name)
            elif self._device_shown is not None:
                logger.warning('Camera device %s gone: only the image source is available', self._device_shown)
            self._device_shown = name
            self.settings.device = name

    def _emit(self, frame: CameraFrame) -> None:
        with self._mutex:
            self._frame = frame
        self._callbacks(frame)

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
        self._image = image

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
