"""A grab of the main screen for SMILE Online: the frame of the render as the room sees it, without
the room's overlays (the QR code, the operator's status line), scaled down and read back to the CPU
at a few frames a second, so the phone can show what the screen shows. The render thread asks `due`
once per frame and, when it is, blits the texture into a small FBO and reads it (`grab`, from
SmileRender.update); another thread takes the newest frame (`take`) and encodes it. Nothing in
modules/gl changes."""

from threading import Lock

import numpy as np
from OpenGL.GL import GL_RGBA8  # type: ignore

from modules.gl import Fbo, Texture
from modules.render.layers.passes import copy


class ScreenGrab:
    def __init__(self) -> None:
        self._fbo: Fbo = Fbo()
        self._mutex: Lock = Lock()
        self._on: bool = False
        self._width: int = 512
        self._interval: float = 0.1
        self._last: float = 0.0
        self._frame: np.ndarray | None = None    # the newest frame, RGBA uint8, upright; None once taken

    # ANY THREAD
    def set(self, on: bool, width: int = 512, fps: float = 10.0) -> None:
        with self._mutex:
            self._on = on
            self._width = max(16, int(width))
            self._interval = 1.0 / max(0.1, fps)
            if not on:
                self._frame = None

    def take(self) -> np.ndarray | None:
        """The newest frame since the last take, or None."""
        with self._mutex:
            frame, self._frame = self._frame, None
        return frame

    # RENDER THREAD
    def due(self, now: float) -> bool:
        """Whether a frame is wanted now (on, and the interval passed); marks the time when it is."""
        with self._mutex:
            on, interval = self._on, self._interval
        if not on or now - self._last < interval:
            return False
        self._last = now
        return True

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

    def deallocate(self) -> None:
        self._fbo.deallocate()
