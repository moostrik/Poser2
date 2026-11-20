"""ScreenPreview: a held screen frame (its YUV planes, in RAM) drawn back as an RGB picture, without the render thread
waiting.

`set_screen_frame` stores the latest frame to show; the preview's own thread copies it straight into one of two
unpack pixel buffers, mapped once on the render thread (persistent and coherent), and marks it written. `update`, on
the render tick, uploads a written buffer into the planes' textures, turns them back into RGB in `fbo`, and sets
a fence on the buffer; once that fence has passed, checked on a later tick without waiting, the buffer is free
again. Only the render thread makes GL calls. See modules/video/VIDEO.md.
"""

import ctypes
import logging
from collections import deque
from threading import Condition, Thread
from typing import cast

import numpy as np
from OpenGL.GL import (GL_ALREADY_SIGNALED, GL_CONDITION_SATISFIED, GL_MAP_COHERENT_BIT, GL_MAP_PERSISTENT_BIT,
                       GL_MAP_WRITE_BIT, GL_PIXEL_UNPACK_BUFFER, GL_RGBA8,
                       GL_SYNC_GPU_COMMANDS_COMPLETE, GL_TEXTURE_2D, GL_UNPACK_ALIGNMENT,
                       glBindBuffer, glBindTexture, glBufferStorage, glClientWaitSync, glDeleteBuffers, glDeleteSync,
                       glFenceSync, glGenBuffers, glMapBufferRange, glPixelStorei, glTexSubImage2D, glUnmapBuffer)

from modules.gl import Fbo, Style
from modules.video import ScreenFormat

from .screen_capture import yuv_planes
from .shaders import YuvToRgb

logger = logging.getLogger(__name__)

BUFFERS: int = 2            # one written by the preview's thread while the other is uploaded
_UNPACK_ALIGNMENT: int = 4  # GL's default, which nothing else in the apps changes: set back after the uploads
_MAP_FLAGS: int = int(GL_MAP_WRITE_BIT) | int(GL_MAP_PERSISTENT_BIT) | int(GL_MAP_COHERENT_BIT)


class ScreenPreview:
    """A held screen frame (its YUV planes, as captured), back as RGB in `fbo`, upright; the latest frame set is the
    one shown."""

    def __init__(self) -> None:
        self.format: ScreenFormat | None = None
        self.fbo: Fbo = Fbo()
        self._planes = yuv_planes(ScreenFormat(2, 2))   # the textures the planes are uploaded into, remade at allocate
        self._shader: YuvToRgb = YuvToRgb()
        self._buffers: list[int] = []
        self._arrays: list[np.ndarray] = []         # each buffer's mapped memory, the format's shape and dtype
        self._wake: Condition = Condition()         # guards what follows
        self._free: deque[int] = deque()
        self._written: deque[tuple[int, int]] = deque()     # (buffer, key), oldest first
        self._wanted: tuple[int, np.ndarray] | None = None  # the latest frame set, not yet copied
        self._exit: bool = False
        self._in_flight: deque[tuple[int, object]] = deque()    # (buffer, fence) of the uploads
        self.shown: int | None = None               # the key of the frame `fbo` holds
        self._thread: Thread | None = None

    @property
    def allocated(self) -> bool:
        return self.fbo.allocated

    @property
    def width(self) -> int:
        return self.format.width if self.format else 0

    @property
    def height(self) -> int:
        return self.format.height if self.format else 0

    # RENDER THREAD

    def allocate(self, screen_format: ScreenFormat) -> None:
        """GL objects for frames of `screen_format` and the two buffers, mapped once, and the thread."""
        self.deallocate()
        self.format = screen_format
        width, height = screen_format.width, screen_format.height
        size = screen_format.frame_bytes
        try:
            self.fbo.allocate(width, height, GL_RGBA8)
            self._planes = yuv_planes(screen_format)
            for plane in self._planes:
                plane.fbo.allocate(plane.width, plane.height, plane.internal_format)
            self._shader.allocate()
            self._buffers = [int(b) for b in np.atleast_1d(glGenBuffers(BUFFERS))]
            for buffer in self._buffers:
                glBindBuffer(GL_PIXEL_UNPACK_BUFFER, buffer)
                glBufferStorage(GL_PIXEL_UNPACK_BUFFER, size, None, _MAP_FLAGS)
                pointer = cast(int | ctypes.c_void_p, glMapBufferRange(GL_PIXEL_UNPACK_BUFFER, 0, size, _MAP_FLAGS))     # the stub says None
                address = pointer if isinstance(pointer, int) else ctypes.cast(pointer, ctypes.c_void_p).value
                if not address:
                    raise RuntimeError('a pixel buffer could not be mapped')
                data = (ctypes.c_uint8 * size).from_address(address)
                self._arrays.append(np.ctypeslib.as_array(data).view(screen_format.dtype).reshape(screen_format.shape))
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)
        except Exception:
            self.deallocate()
            raise
        with self._wake:
            self._free = deque(range(BUFFERS))
            self._written.clear()
            self._wanted = None
            self._exit = False
        self.shown = None
        self._thread = Thread(target=self._run, name='screen preview', daemon=True)
        self._thread.start()

    def deallocate(self) -> None:
        """Stop the thread, then free the buffers and the GL objects."""
        if self._thread is not None:
            with self._wake:
                self._exit = True
                self._wake.notify()
            self._thread.join()
            self._thread = None
        for _, fence in self._in_flight:
            glDeleteSync(fence)
        self._in_flight.clear()
        for buffer in self._buffers:
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, buffer)
            glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER)
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)
        self._arrays = []
        if self._buffers:
            glDeleteBuffers(len(self._buffers), self._buffers)
        self._buffers = []
        if self._shader.allocated:
            self._shader.deallocate()
        for texture in [self.fbo] + [plane.fbo for plane in self._planes]:
            if texture.allocated:
                texture.deallocate()
        self.shown = None
        self.format = None

    def update(self) -> None:
        """Free the buffers whose upload has passed its fence, then upload the newest written one and draw it, without
        waiting."""
        while self._in_flight:
            buffer, fence = self._in_flight[0]
            if glClientWaitSync(fence, 0, 0) not in (GL_ALREADY_SIGNALED, GL_CONDITION_SATISFIED):
                break
            self._in_flight.popleft()
            glDeleteSync(fence)
            with self._wake:
                self._free.append(buffer)
                self._wake.notify()
        with self._wake:
            if not self._written:
                return
            while len(self._written) > 1:           # only the newest is drawn; the older go back
                self._free.append(self._written.popleft()[0])
            buffer, key = self._written.popleft()
            self._wake.notify()
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, self._buffers[buffer])
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)   # rows of any even width, unpadded
        for plane in self._planes:
            glBindTexture(GL_TEXTURE_2D, plane.fbo.tex_id)
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, plane.width, plane.height, plane.read_format, plane.sample_type,
                            ctypes.c_void_p(plane.offset))
        glBindTexture(GL_TEXTURE_2D, 0)
        glPixelStorei(GL_UNPACK_ALIGNMENT, _UNPACK_ALIGNMENT)
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)
        self._in_flight.append((buffer, glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0)))
        Style.push_style()
        Style.set_blend_mode(Style.BlendMode.DISABLED)
        self.fbo.begin()
        self._shader.use([plane.fbo for plane in self._planes], self._planes[0].sample_max)
        self.fbo.end()
        Style.pop_style()
        self.shown = key

    # ANY THREAD

    def set_screen_frame(self, key: int, data: np.ndarray) -> None:
        """Store the frame to show, in the format's layout, under `key`; a frame nobody changes while it is shown,
        as a held slot. Only the latest frame set is shown."""
        with self._wake:
            if self.shown == key or (self._wanted is not None and self._wanted[0] == key):
                return
            if any(written_key == key for _, written_key in self._written):
                return
            self._wanted = (key, data)
            self._wake.notify()

    # THE PREVIEW'S THREAD: MAPPED MEMORY ONLY, NO GL

    def _run(self) -> None:
        while True:
            with self._wake:
                while not self._exit and (self._wanted is None or not self._free):
                    self._wake.wait()
                if self._exit:
                    return
                wanted = self._wanted
                assert wanted is not None
                self._wanted = None
                buffer = self._free.popleft()
            key, data = wanted
            try:
                np.copyto(self._arrays[buffer], data)
            except Exception as e:
                logger.error('Screen preview could not copy a frame: %s', e)
                with self._wake:
                    self._free.append(buffer)
                continue
            with self._wake:
                self._written.append((buffer, key))
