"""ScreenCapture: the recording's screen frame, converted to NV12, read back into RAM and handed over when it is done.

The render thread draws into `frame` (or copies the window into it with `copy_window`) and calls `capture`; `poll`
on every tick hands over each frame whose read-back has passed its fence. The pixel buffers are host memory mapped
once, so the read-back needs no CUDA and the CPU never waits for it. Only the render thread makes GL calls; the
consumer reads a lent buffer on its own thread and gives it back with the frame's `release`. See
modules/video/VIDEO.md.
"""

import ctypes
import logging
from collections import deque
from functools import partial
from threading import Lock

import numpy as np
from OpenGL.GL import (GL_ALREADY_SIGNALED, GL_COLOR_BUFFER_BIT, GL_CONDITION_SATISFIED, GL_DRAW_FRAMEBUFFER,
                       GL_FRAMEBUFFER, GL_LINEAR, GL_MAP_COHERENT_BIT, GL_MAP_PERSISTENT_BIT, GL_MAP_READ_BIT,
                       GL_NEAREST, GL_PACK_ALIGNMENT, GL_PIXEL_PACK_BUFFER, GL_R8, GL_READ_FRAMEBUFFER, GL_RED, GL_RG,
                       GL_RG8, GL_RGBA8, GL_SYNC_GPU_COMMANDS_COMPLETE, GL_UNSIGNED_BYTE, glBindBuffer,
                       glBindFramebuffer, glBlitFramebuffer, glBufferStorage, glClear, glClearColor, glClientWaitSync,
                       glDeleteBuffers, glDeleteSync, glFenceSync, glGenBuffers, glMapBufferRange, glPixelStorei,
                       glReadPixels, glUnmapBuffer)

from modules.gl import Fbo, Style, fill, fit
from modules.video import ScreenFrame, ScreenFrameCallback

from .shaders import Nv12, Nv12Plane

logger = logging.getLogger(__name__)

POOL: int = 4               # pixel buffers: one being read back, the rest with the recorder or waiting for it
_PACK_ALIGNMENT: int = 4    # GL's default, which nothing else in the apps changes: set back after the capture's reads
_MAP_FLAGS: int = GL_MAP_READ_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT


def nv12_bytes(width: int, height: int) -> int:
    """Bytes of one NV12 frame: the Y plane, then Cb Cr interleaved at half width and half height."""
    return width * height * 3 // 2


class ScreenCapture:
    """The screen frame (RGBA8) of the recording's size, its NV12 conversion, and a pool of four pixel buffers in host
    memory the conversion is read back into; each finished frame goes to the callbacks, lent until it is released."""

    def __init__(self) -> None:
        self.width: int = 0
        self.height: int = 0
        self.frame: Fbo = Fbo()
        self._y: Fbo = Fbo(GL_NEAREST)
        self._uv: Fbo = Fbo(GL_NEAREST)
        self._shader: Nv12 = Nv12()
        self._buffers: list[int] = []
        self._arrays: list[np.ndarray] = []         # each buffer's mapped memory, (height × 3/2, width) uint8
        self._in_flight: deque[tuple[int, object, float, int]] = deque()     # (buffer, fence, timestamp, skipped)
        self._lock: Lock = Lock()                   # guards the free list and the lent count
        self._free: deque[int] = deque()
        self._lent: int = 0
        self._skipped: int = 0                      # ticks left out since the last frame went out
        self._callbacks: list[ScreenFrameCallback] = []
        self._callback_lock: Lock = Lock()

    @property
    def allocated(self) -> bool:
        return self.frame.allocated

    @property
    def lent(self) -> bool:
        """Whether a buffer is with a consumer or a read-back still on its way: the pool cannot be freed then."""
        with self._lock:
            return self._lent > 0 or bool(self._in_flight)

    def add_screen_frame_callback(self, callback: ScreenFrameCallback) -> None:
        with self._callback_lock:
            if callback not in self._callbacks:
                self._callbacks.append(callback)

    def remove_screen_frame_callback(self, callback: ScreenFrameCallback) -> None:
        with self._callback_lock:
            if callback in self._callbacks:
                self._callbacks.remove(callback)

    # RENDER THREAD

    def allocate(self, width: int, height: int) -> None:
        """GL objects for frames of `width` × `height` (even) and the pool, mapped once; raises on failure with nothing
        left allocated, or when a buffer is still lent."""
        if width <= 0 or height <= 0 or width % 2 or height % 2:
            raise ValueError(f'NV12 needs an even size, got {width}x{height}')
        self.deallocate()
        self.width, self.height = width, height
        size = nv12_bytes(width, height)
        try:
            self.frame.allocate(width, height, GL_RGBA8)
            self._y.allocate(width, height, GL_R8)
            self._uv.allocate(width // 2, height // 2, GL_RG8)
            self._shader.allocate()
            self._buffers = [int(b) for b in np.atleast_1d(glGenBuffers(POOL))]
            for buffer in self._buffers:
                glBindBuffer(GL_PIXEL_PACK_BUFFER, buffer)
                glBufferStorage(GL_PIXEL_PACK_BUFFER, size, None, _MAP_FLAGS)
                pointer = glMapBufferRange(GL_PIXEL_PACK_BUFFER, 0, size, _MAP_FLAGS)
                address = pointer if isinstance(pointer, int) else ctypes.cast(pointer, ctypes.c_void_p).value
                if not address:
                    raise RuntimeError('a pixel buffer could not be mapped')
                data = (ctypes.c_uint8 * size).from_address(address)
                self._arrays.append(np.ctypeslib.as_array(data).reshape(height * 3 // 2, width))
            glBindBuffer(GL_PIXEL_PACK_BUFFER, 0)
            with self._lock:
                self._free = deque(range(POOL))
                self._skipped = 0
        except Exception:
            self.deallocate()
            raise

    def deallocate(self) -> None:
        """Free the GL objects and the pool; raises when a buffer is still lent, as its memory would go under a
        reader."""
        with self._lock:
            if self._lent:
                raise RuntimeError('a screen frame is still lent')
        for _, fence, _, _ in self._in_flight:
            glDeleteSync(fence)
        self._in_flight.clear()
        for buffer in self._buffers:
            glBindBuffer(GL_PIXEL_PACK_BUFFER, buffer)
            glUnmapBuffer(GL_PIXEL_PACK_BUFFER)
        glBindBuffer(GL_PIXEL_PACK_BUFFER, 0)
        self._arrays = []
        if self._buffers:
            glDeleteBuffers(len(self._buffers), self._buffers)
        self._buffers = []
        with self._lock:
            self._free.clear()
        if self._shader.allocated:
            self._shader.deallocate()
        for texture in (self.frame, self._y, self._uv):
            if texture.allocated:
                texture.deallocate()
        self.width = self.height = 0

    def copy_window(self, window_width: int, window_height: int, fill_frame: bool = False) -> None:
        """Copy the window's back buffer into `frame`, fit with black bars, or filling it with the overhang cut."""
        place = fill if fill_frame else fit
        x, y, w, h = place(window_width, window_height, self.width, self.height)
        self.frame.begin()
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)
        glBindFramebuffer(GL_READ_FRAMEBUFFER, 0)
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, self.frame.fbo_id)
        glBlitFramebuffer(0, 0, window_width, window_height,
                          round(x), round(y), round(x + w), round(y + h), GL_COLOR_BUFFER_BIT, GL_LINEAR)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    def capture(self, timestamp: float) -> None:
        """Convert `frame` and read it back into a free pixel buffer, with a fence after; with none free the tick is
        left out and counted on the next frame."""
        with self._lock:
            buffer = self._free.popleft() if self._free else None
            if buffer is None:
                self._skipped += 1
                return
            skipped, self._skipped = self._skipped, 0
        Style.push_style()
        Style.set_blend_mode(Style.BlendMode.DISABLED)
        self._y.begin()
        self._shader.use(self.frame, Nv12Plane.Y)
        self._uv.begin()
        self._shader.use(self.frame, Nv12Plane.UV)
        Style.pop_style()
        glPixelStorei(GL_PACK_ALIGNMENT, 1)     # rows of any even width, unpadded
        glBindBuffer(GL_PIXEL_PACK_BUFFER, self._buffers[buffer])
        glBindFramebuffer(GL_READ_FRAMEBUFFER, self._y.fbo_id)
        glReadPixels(0, 0, self.width, self.height, GL_RED, GL_UNSIGNED_BYTE, 0)
        glBindFramebuffer(GL_READ_FRAMEBUFFER, self._uv.fbo_id)
        glReadPixels(0, 0, self.width // 2, self.height // 2, GL_RG, GL_UNSIGNED_BYTE, self.width * self.height)
        glBindBuffer(GL_PIXEL_PACK_BUFFER, 0)
        glPixelStorei(GL_PACK_ALIGNMENT, _PACK_ALIGNMENT)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        fence = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0)
        self._in_flight.append((buffer, fence, timestamp, skipped))

    def poll(self) -> None:
        """Hand over each read-back whose fence has passed, oldest first, without waiting for the others."""
        while self._in_flight:
            buffer, fence, timestamp, skipped = self._in_flight[0]
            if glClientWaitSync(fence, 0, 0) not in (GL_ALREADY_SIGNALED, GL_CONDITION_SATISFIED):
                return
            self._in_flight.popleft()
            glDeleteSync(fence)
            with self._lock:
                self._lent += 1
            frame = ScreenFrame(self._arrays[buffer], timestamp, skipped, partial(self._give_back, buffer))
            with self._callback_lock:
                callbacks = list(self._callbacks)
            if not callbacks:
                frame.release()
            for callback in callbacks:
                callback(frame)

    # ANY THREAD

    def _give_back(self, buffer: int) -> None:
        with self._lock:
            self._lent -= 1
            self._free.append(buffer)
