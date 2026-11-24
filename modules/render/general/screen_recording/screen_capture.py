"""ScreenCapture: the recording's screen frame, converted to YUV planes, read back into RAM and handed over when it is done.
See SCREEN_RECORDING.md.
"""

import logging
from collections import deque
from functools import partial
from threading import Lock

import numpy as np
from OpenGL.GL import (GL_COLOR_BUFFER_BIT, GL_DRAW_FRAMEBUFFER, GL_FRAMEBUFFER, GL_LINEAR, GL_NEAREST,
                       GL_PACK_ALIGNMENT, GL_PIXEL_PACK_BUFFER, GL_R16, GL_R8, GL_READ_FRAMEBUFFER, GL_RED, GL_RG,
                       GL_RG16, GL_RG8, GL_RGB10_A2, GL_RGBA, GL_RGBA16F, GL_RGBA8, GL_UNSIGNED_BYTE,
                       GL_UNSIGNED_INT_2_10_10_10_REV, GL_UNSIGNED_SHORT, glBindFramebuffer, glBlitFramebuffer,
                       glClear, glClearColor, glPixelStorei, glReadPixels)

from modules.gl import BlendState, Fbo, PixelBufferPool
from modules.utils import Broadcast, fill, fit
from modules.video import BitDepth, Chroma, ScreenFormat, ScreenFrame, ScreenFrameCallback

from .shaders import RgbToYuv, YuvPlane

logger = logging.getLogger(__name__)

POOL: int = 4               # pixel buffers: one being read back, the rest with the recorder or waiting for it


class _Plane:
    """One YUV plane of the frame: the Fbo it is drawn into, which plane, its GL formats and sample type, the
    largest value a sample holds, and where it lies in the read-back (bytes)."""

    def __init__(self, plane: YuvPlane, width: int, height: int, internal_format: int, read_format: int,
                 sample_type: int, sample_max: int, offset: int) -> None:
        self.fbo: Fbo = Fbo(GL_NEAREST)
        self.plane: YuvPlane = plane
        self.width, self.height = width, height
        self.internal_format, self.read_format, self.sample_type = internal_format, read_format, sample_type
        self.sample_max: int = sample_max
        self.offset: int = offset


def yuv_planes(screen_format: ScreenFormat) -> list[_Plane]:
    """The planes of a format, in the order the frame holds them: Y, then Cb Cr (4:2:0) or Cb and Cr (4:4:4); 8-bit
    samples in bytes, 4:2:0 at 10 bits in 16-bit textures, 4:4:4 at 10 bits packed in one RGB10_A2 plane."""
    width, height = screen_format.width, screen_format.height
    if screen_format.packed:
        return [_Plane(YuvPlane.CBYCR, width, height, GL_RGB10_A2, GL_RGBA, GL_UNSIGNED_INT_2_10_10_10_REV, 1023, 0)]
    eight = screen_format.bit_depth is BitDepth.BIT_8
    one, two = (GL_R8, GL_RG8) if eight else (GL_R16, GL_RG16)
    sample_type, sample_max = (GL_UNSIGNED_BYTE, 255) if eight else (GL_UNSIGNED_SHORT, 65535)
    plane_bytes = width * height * np.dtype(screen_format.dtype).itemsize
    planes = [_Plane(YuvPlane.Y, width, height, one, GL_RED, sample_type, sample_max, 0)]
    if screen_format.chroma is Chroma.YUV420:
        planes.append(_Plane(YuvPlane.CBCR, width // 2, height // 2, two, GL_RG, sample_type, sample_max, plane_bytes))
    else:
        planes.append(_Plane(YuvPlane.CB, width, height, one, GL_RED, sample_type, sample_max, plane_bytes))
        planes.append(_Plane(YuvPlane.CR, width, height, one, GL_RED, sample_type, sample_max, 2 * plane_bytes))
    return planes


class ScreenCapture:
    """The screen frame of the recording's size (RGBA8, or RGBA16F at 10 bits), its YUV planes, and a pool of four
    pixel buffers in host memory they are read back into; each finished frame goes to the callbacks, lent until it is released."""

    def __init__(self) -> None:
        self.format: ScreenFormat | None = None
        self.frame: Fbo = Fbo()
        self._planes: list[_Plane] = []
        self._shader: RgbToYuv = RgbToYuv()
        self._pixels: PixelBufferPool = PixelBufferPool(GL_PIXEL_PACK_BUFFER)    # the pool, mapped once
        self._arrays: list[np.ndarray] = []         # each buffer's mapped memory, the format's shape and dtype
        self._in_flight: deque[tuple[int, float, int]] = deque()     # (buffer, timestamp, dropped), oldest first
        self._lock: Lock = Lock()                   # guards the free list and the lent count
        self._free: deque[int] = deque()
        self._lent: int = 0
        self._dropped: int = 0                      # ticks left out since the last frame went out
        self._callbacks: Broadcast[ScreenFrame] = Broadcast(name="screen frame")

    @property
    def allocated(self) -> bool:
        return self.frame.allocated

    @property
    def width(self) -> int:
        return self.format.width if self.format else 0

    @property
    def height(self) -> int:
        return self.format.height if self.format else 0

    @property
    def lent(self) -> bool:
        """Whether a buffer is with a consumer or a read-back still on its way: the pool cannot be freed then."""
        with self._lock:
            return self._lent > 0 or bool(self._in_flight)

    def add_screen_frame_callback(self, callback: ScreenFrameCallback) -> None:
        self._callbacks.add_callback(callback)

    def remove_screen_frame_callback(self, callback: ScreenFrameCallback) -> None:
        self._callbacks.remove_callback(callback)

    # MAIN THREAD

    def allocate(self, screen_format: ScreenFormat) -> None:
        """GL objects for frames of `screen_format` (an even size) and the pool, mapped once; raises on failure with
        nothing left allocated, or when a buffer is still lent."""
        width, height = screen_format.width, screen_format.height
        if width <= 0 or height <= 0 or width % 2 or height % 2:
            raise ValueError(f'YUV planes need an even size, got {width}x{height}')
        self.deallocate()
        self.format = screen_format
        size = screen_format.frame_bytes
        try:
            self.frame.allocate(width, height, GL_RGBA8 if screen_format.bit_depth is BitDepth.BIT_8 else GL_RGBA16F)
            self._planes = yuv_planes(screen_format)
            for plane in self._planes:
                plane.fbo.allocate(plane.width, plane.height, plane.internal_format)
            self._shader.allocate()
            self._pixels.allocate(POOL, size, mapped=True)
            self._arrays = [self._pixels.memory(slot).view(screen_format.dtype).reshape(screen_format.shape)
                            for slot in range(POOL)]
            with self._lock:
                self._free = deque(range(POOL))
                self._dropped = 0
        except Exception:
            self.deallocate()
            raise

    def deallocate(self) -> None:
        """Free the GL objects and the pool; raises when a buffer is still lent."""
        with self._lock:
            if self._lent:
                raise RuntimeError('a screen frame is still lent')
        self._in_flight.clear()
        self._arrays = []
        self._pixels.deallocate()
        with self._lock:
            self._free.clear()
        if self._shader.allocated:
            self._shader.deallocate()
        for texture in [self.frame] + [plane.fbo for plane in self._planes]:
            if texture.allocated:
                texture.deallocate()
        self._planes = []
        self.format = None

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
                self._dropped += 1
                return
            dropped, self._dropped = self._dropped, 0
        BlendState.push()
        BlendState.set_blend_mode(BlendState.BlendMode.DISABLED)
        for plane in self._planes:
            plane.fbo.begin()
            self._shader.use(self.frame, plane.plane, plane.sample_max)
        BlendState.pop()
        glPixelStorei(GL_PACK_ALIGNMENT, 1)     # rows of any width, unpadded
        self._pixels.bind(buffer)
        for plane in self._planes:
            glBindFramebuffer(GL_READ_FRAMEBUFFER, plane.fbo.fbo_id)
            glReadPixels(0, 0, plane.width, plane.height, plane.read_format, plane.sample_type, plane.offset)
        self._pixels.unbind()
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        self._pixels.fence(buffer)
        self._in_flight.append((buffer, timestamp, dropped))

    def poll(self) -> None:
        """Hand over each read-back whose fence has passed, oldest first, without waiting for the others."""
        while self._in_flight:
            buffer, timestamp, dropped = self._in_flight[0]
            if not self._pixels.passed(buffer):
                return
            self._in_flight.popleft()
            with self._lock:
                self._lent += 1
            frame = ScreenFrame(self._arrays[buffer], timestamp, dropped, partial(self._give_back, buffer))
            if self._callbacks:
                self._callbacks(frame)
            else:
                frame.release()

    # ANY THREAD

    def _give_back(self, buffer: int) -> None:
        with self._lock:
            self._lent -= 1
            self._free.append(buffer)
