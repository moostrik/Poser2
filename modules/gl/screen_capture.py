"""ScreenCapture: the recording's screen frame, converted to NV12 and copied into a ring of frames in CUDA memory.

The render thread draws into `frame` (or copies the window into it with `copy_window`) and calls `capture`. The
ring's blocks are plain `cudaMalloc` memory, added and freed by any thread; `download` reads a frame back into
pinned host memory on a stream of its own. Only the render thread makes GL calls. See modules/video/VIDEO.md.
"""

import ctypes
import logging
from collections import deque
from threading import Lock

import numpy as np
from OpenGL.GL import (GL_BLEND, GL_COLOR_BUFFER_BIT, GL_DRAW_FRAMEBUFFER, GL_FRAMEBUFFER, GL_LINEAR, GL_NEAREST,
                       GL_PACK_ALIGNMENT, GL_PIXEL_PACK_BUFFER, GL_R8, GL_READ_FRAMEBUFFER, GL_RED, GL_RG, GL_RG8,
                       GL_RGBA8, GL_STREAM_COPY, GL_UNSIGNED_BYTE, glBindBuffer, glBindFramebuffer, glBlitFramebuffer,
                       glBufferData, glClear, glClearColor, glDeleteBuffers, glDisable, glEnable, glGenBuffers,
                       glGetIntegerv, glIsEnabled, glPixelStorei, glReadPixels)

try:
    from . import cudart   # binds torch's CUDA runtime; fails without a CUDA torch
    _CUDA_OK = True
except (ImportError, OSError):
    _CUDA_OK = False
from .fbo import Fbo
from .shaders import Nv12, Nv12Plane
from .utils import fill, fit

logger = logging.getLogger(__name__)

PIXEL_BUFFERS: int = 2      # one written by GL this tick, one copied by CUDA from last tick


def nv12_bytes(width: int, height: int) -> int:
    """Bytes of one NV12 frame: the Y plane, then Cb Cr interleaved at half width and half height."""
    return width * height * 3 // 2


class ScreenCapture:
    """The screen frame (RGBA8) of the recording's size, its NV12 conversion, and the ring of NV12 frames in CUDA
    memory it is copied into, one tick late."""

    def __init__(self) -> None:
        self.width: int = 0
        self.height: int = 0
        self.frame: Fbo = Fbo()
        self._y: Fbo = Fbo(GL_NEAREST)
        self._uv: Fbo = Fbo(GL_NEAREST)
        self._shader: Nv12 = Nv12()
        self._pixel_buffers: list[int] = []
        self._resources: list[int] = []
        self._turn: int = 0
        self._pending: int | None = None    # the ring slot of the frame in the other pixel buffer
        self._capture_stream: int = 0
        self._download_stream: int = 0
        self._lock: Lock = Lock()           # guards the block list
        self._blocks: list[tuple[int, int]] = []    # (device pointer, slots)
        self._host: list[int] = []          # pinned download buffers
        self._host_events: list[int] = []   # one per download buffer: its copy is done
        self._downloads: deque[int] = deque()   # host buffer indices queued, oldest first
        self._next_host: int = 0

    @property
    def allocated(self) -> bool:
        return self.frame.allocated

    @property
    def frame_bytes(self) -> int:
        return nv12_bytes(self.width, self.height)

    # RENDER THREAD

    def allocate(self, width: int, height: int) -> None:
        """GL objects for frames of `width` × `height` (even), and their CUDA registration; raises on failure, with
        nothing left allocated."""
        if width <= 0 or height <= 0 or width % 2 or height % 2:
            raise ValueError(f'NV12 needs an even size, got {width}x{height}')
        if not _CUDA_OK:
            raise OSError('screen capture needs torch with CUDA')
        self.deallocate()
        self.width, self.height = width, height
        try:
            self.frame.allocate(width, height, GL_RGBA8)
            self._y.allocate(width, height, GL_R8)
            self._uv.allocate(width // 2, height // 2, GL_RG8)
            self._shader.allocate()
            self._pixel_buffers = [int(b) for b in np.atleast_1d(glGenBuffers(PIXEL_BUFFERS))]
            for buffer in self._pixel_buffers:
                glBindBuffer(GL_PIXEL_PACK_BUFFER, buffer)
                # COPY: written and read by the GPU only, so the driver keeps it in video memory
                glBufferData(GL_PIXEL_PACK_BUFFER, self.frame_bytes, None, GL_STREAM_COPY)
            glBindBuffer(GL_PIXEL_PACK_BUFFER, 0)
            for buffer in self._pixel_buffers:
                self._resources.append(cudart.register_gl_buffer(buffer, cudart.REGISTER_FLAGS_READ_ONLY))
            self._capture_stream = cudart.stream_create()
            self._download_stream = cudart.stream_create()
        except Exception:
            self.deallocate()
            raise

    def deallocate(self) -> None:
        """Free the GL objects, their registration and the streams. The ring's blocks are freed by `free_blocks`."""
        if self._capture_stream:
            cudart.stream_synchronize(self._capture_stream)
        for resource in self._resources:
            try:
                cudart.unregister_resource(resource)
            except cudart.CudaError as e:
                logger.warning('Screen capture unregister failed: %s', e)
        self._resources = []
        if self._pixel_buffers:
            glDeleteBuffers(len(self._pixel_buffers), self._pixel_buffers)
        self._pixel_buffers = []
        for stream in (self._capture_stream, self._download_stream):
            if stream:
                cudart.stream_destroy(stream)
        self._capture_stream = self._download_stream = 0
        if self._shader.allocated:
            self._shader.deallocate()
        for fbo in (self.frame, self._y, self._uv):
            if fbo.allocated:
                fbo.deallocate()
        self._pending = None
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

    def capture(self, slot: int) -> None:
        """Copy last tick's frame into its ring slot, and convert `frame` into the other pixel buffer, to go into
        `slot` on the next `capture` or `flush`."""
        self.flush()
        buffer = self._pixel_buffers[self._turn]
        blend = glIsEnabled(GL_BLEND)
        glDisable(GL_BLEND)
        self._y.begin()
        self._shader.use(self.frame, Nv12Plane.Y)
        self._uv.begin()
        self._shader.use(self.frame, Nv12Plane.UV)
        if blend:
            glEnable(GL_BLEND)
        alignment = int(glGetIntegerv(GL_PACK_ALIGNMENT))
        glPixelStorei(GL_PACK_ALIGNMENT, 1)     # rows of any even width, unpadded
        glBindBuffer(GL_PIXEL_PACK_BUFFER, buffer)
        glBindFramebuffer(GL_READ_FRAMEBUFFER, self._y.fbo_id)
        glReadPixels(0, 0, self.width, self.height, GL_RED, GL_UNSIGNED_BYTE, 0)
        glBindFramebuffer(GL_READ_FRAMEBUFFER, self._uv.fbo_id)
        glReadPixels(0, 0, self.width // 2, self.height // 2, GL_RG, GL_UNSIGNED_BYTE, self.width * self.height)
        glBindBuffer(GL_PIXEL_PACK_BUFFER, 0)
        glPixelStorei(GL_PACK_ALIGNMENT, alignment)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        self._pending = slot
        self._turn = 1 - self._turn

    def flush(self) -> None:
        """Copy the pending frame, written by GL on an earlier tick, into its slot."""
        if self._pending is None:
            return
        slot, self._pending = self._pending, None
        resource = self._resources[1 - self._turn]
        pointer = self.slot_pointer(slot)
        cudart.map_resource(resource, self._capture_stream)
        try:
            source, _ = cudart.mapped_pointer(resource)
            cudart.memcpy_device_to_device(pointer, source, self.frame_bytes, self._capture_stream)
        finally:
            cudart.unmap_resource(resource, self._capture_stream)

    # ANY THREAD: CUDA ONLY

    @property
    def num_slots(self) -> int:
        with self._lock:
            return sum(slots for _, slots in self._blocks)

    def slot_pointer(self, slot: int) -> int:
        """The device pointer of ring slot `slot`, counted over the blocks in the order they were added."""
        with self._lock:
            for pointer, slots in self._blocks:
                if slot < slots:
                    return pointer + slot * self.frame_bytes
                slot -= slots
        raise IndexError('screen capture slot outside the ring')

    @staticmethod
    def free_memory() -> int:
        """The device's free memory in bytes."""
        return cudart.mem_get_info()[0]

    def add_block(self, slots: int) -> None:
        """Grow the ring by a block of `slots` frames; raises `cudart.CudaError` when the device has no room."""
        pointer = cudart.malloc(slots * self.frame_bytes)
        with self._lock:
            self._blocks.append((pointer, slots))

    def free_blocks(self) -> None:
        """Free the ring's blocks and the download buffers; the memory goes back to the device."""
        self.synchronize()
        with self._lock:
            blocks, self._blocks = self._blocks, []
        for pointer, _ in blocks:
            cudart.free(pointer)
        if self._download_stream:
            cudart.stream_synchronize(self._download_stream)
        for pointer in self._host:
            cudart.free_host(pointer)
        for event in self._host_events:
            cudart.event_destroy(event)
        self._host = []
        self._host_events = []
        self._downloads.clear()

    def synchronize(self) -> None:
        """Wait until every queued copy into the ring is done."""
        if self._capture_stream:
            cudart.stream_synchronize(self._capture_stream)

    def queue_download(self, slot: int) -> None:
        """Queue the copy of `slot` into the next of two pinned host buffers; `wait_download` returns it. With one
        download queued, the next copy runs while the caller writes out the one before."""
        if len(self._downloads) >= 2:
            raise RuntimeError('two downloads are queued already')
        if not self._host:
            self._host = [cudart.malloc_host(self.frame_bytes) for _ in range(2)]
            self._host_events = [cudart.event_create() for _ in range(2)]
        index = self._next_host
        self._next_host = 1 - index
        cudart.memcpy_device_to_host(self._host[index], self.slot_pointer(slot), self.frame_bytes, self._download_stream)
        cudart.event_record(self._host_events[index], self._download_stream)
        self._downloads.append(index)

    def wait_download(self) -> np.ndarray:
        """The oldest queued download as a (height × 3/2, width) uint8 NV12 array, valid until the next
        `queue_download` after this call."""
        index = self._downloads.popleft()
        cudart.event_synchronize(self._host_events[index])
        data = (ctypes.c_uint8 * self.frame_bytes).from_address(self._host[index])
        return np.ctypeslib.as_array(data).reshape(self.height * 3 // 2, self.width)
