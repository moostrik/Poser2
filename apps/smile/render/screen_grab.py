"""A grab of the main screen for SMILE Online: the frame of the render as the room sees it, without
the room's overlays (the QR code, the operator's status line), scaled down and read back to the CPU
at a few frames a second, so the phone can show what the screen shows. The render thread asks `due`
once per frame and, when it is, blits the texture into a small FBO and reads it (`grab`, from
SmileRender.update); another thread takes the newest frame (`take`) and encodes it. The turn's last frame
(`last`) is taken at once and without the caption (`clean`, for the render): the phone fades the caption
out itself and keeps that frame. Nothing in modules/gl changes."""

from threading import Lock

import numpy as np
from OpenGL.GL import GL_RGBA8  # type: ignore

from modules.gl import Fbo, Texture
from .layers import copy


class ScreenGrab:
    def __init__(self) -> None:
        self._fbo: Fbo = Fbo()
        self._mutex: Lock = Lock()
        self._on: bool = False
        self._width: int = 512
        self._interval: float = 0.1
        self._time: float = 0.0
        self._frame: np.ndarray | None = None    # the newest frame, RGBA uint8, upright; None once taken
        self._last_wanted: bool = False          # the turn's last frame is asked for: the next render frame, without the caption
        self._last_frame: bool = False           # the frame held is that last one

    # ANY THREAD
    def set(self, on: bool, width: int = 512, fps: float = 10.0) -> None:
        with self._mutex:
            self._on = on
            self._width = max(16, int(width))
            self._interval = 1.0 / max(0.1, fps)
            if not on:
                self._frame = None
                self._last_wanted = False
                self._last_frame = False

    def last(self) -> None:
        """Ask for the turn's last frame: the next render frame, at once, without the caption; `take` marks it."""
        with self._mutex:
            self._last_wanted = True

    def take(self) -> tuple[np.ndarray | None, bool]:
        """The newest frame since the last take (or None), and whether it is the turn's last."""
        with self._mutex:
            frame, self._frame = self._frame, None
            last, self._last_frame = self._last_frame, False
        return frame, last

    # RENDER THREAD
    def due(self, now: float) -> bool:
        """Whether a frame is wanted now (on, and the interval passed, or the last frame asked for); marks the
        time when it is."""
        with self._mutex:
            on, interval, last = self._on, self._interval, self._last_wanted
        if not on or (now - self._time < interval and not last):
            return False
        self._time = now
        return True

    @property
    def clean(self) -> bool:
        """Whether the frame grabbed now is the turn's last: rendered without the caption."""
        with self._mutex:
            return self._last_wanted

    def grab(self, output: Texture) -> None:
        """Read `output` as the newest frame (call when `due`)."""
        with self._mutex:
            width: int = self._width
        if not output.allocated:
            return
        height: int = max(1, round(width * output.height / output.width))
        if not self._fbo.allocated or self._fbo.width != width or self._fbo.height != height:
            self._fbo.deallocate()
            self._fbo.allocate(width, height, GL_RGBA8)
        copy(output, self._fbo)
        frame: np.ndarray | None = self._fbo.read_to_numpy()
        if frame is not None:
            with self._mutex:
                if self._on:
                    self._frame = frame
                    self._last_frame = self._last_wanted
                self._last_wanted = False

    def deallocate(self) -> None:
        self._fbo.deallocate()
