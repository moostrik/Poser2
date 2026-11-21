"""PixelBufferPool: pixel buffers of one size for a readback or an upload, each with the fence of its transfer."""

import ctypes
from typing import cast

import numpy as np
from OpenGL.GL import (GL_ALREADY_SIGNALED, GL_CONDITION_SATISFIED, GL_MAP_COHERENT_BIT, GL_MAP_PERSISTENT_BIT,
                       GL_MAP_READ_BIT, GL_MAP_WRITE_BIT, GL_PIXEL_PACK_BUFFER, GL_STREAM_DRAW, GL_STREAM_READ,
                       GL_SYNC_GPU_COMMANDS_COMPLETE, glBindBuffer, glBufferData, glBufferStorage, glClientWaitSync,
                       glDeleteBuffers, glDeleteSync, glFenceSync, glGenBuffers, glGetBufferSubData, glMapBufferRange,
                       glUnmapBuffer)


class PixelBufferPool:
    """`count` pixel buffers of `size` bytes on `target`: GL_PIXEL_PACK_BUFFER for a readback, GL_PIXEL_UNPACK_BUFFER
    for an upload; mapped once for good (persistent, coherent) or not mapped, then copied out with `read`. Each slot
    holds the fence of the transfer queued into it last. The caller keeps which slots are free and does the transfer
    itself, the slot bound; main thread only, a mapped slot's memory read or written on any."""

    def __init__(self, target: int) -> None:
        self._target: int = target
        self._buffers: list[int] = []
        self._fences: list[object | None] = []
        self._memory: list[np.ndarray] = []      # each mapped buffer's bytes
        self.size: int = 0

    @property
    def allocated(self) -> bool:
        return bool(self._buffers)

    @property
    def count(self) -> int:
        return len(self._buffers)

    def allocate(self, count: int, size: int, mapped: bool = False) -> None:
        """`count` buffers of `size` bytes, mapped when `mapped`; raises with nothing left allocated when a buffer
        cannot be mapped."""
        self.deallocate()
        self.size = size
        readback: bool = self._target == GL_PIXEL_PACK_BUFFER
        flags: int = int(GL_MAP_READ_BIT if readback else GL_MAP_WRITE_BIT) | int(GL_MAP_PERSISTENT_BIT) \
            | int(GL_MAP_COHERENT_BIT)
        try:
            self._buffers = [int(b) for b in np.atleast_1d(glGenBuffers(count))]
            self._fences = [None] * count
            for buffer in self._buffers:
                glBindBuffer(self._target, buffer)
                if not mapped:
                    glBufferData(self._target, size, None, GL_STREAM_READ if readback else GL_STREAM_DRAW)
                    continue
                glBufferStorage(self._target, size, None, flags)
                pointer = cast(int | ctypes.c_void_p, glMapBufferRange(self._target, 0, size, flags))     # the stub says None
                address = pointer if isinstance(pointer, int) else ctypes.cast(pointer, ctypes.c_void_p).value
                if not address:
                    raise RuntimeError('a pixel buffer could not be mapped')
                self._memory.append(np.ctypeslib.as_array((ctypes.c_uint8 * size).from_address(address)))
            glBindBuffer(self._target, 0)
        except Exception:
            glBindBuffer(self._target, 0)
            self.deallocate()
            raise

    def deallocate(self) -> None:
        """The fences dropped, the mapped buffers unmapped, the buffers freed."""
        self.reset()
        if self._memory:
            for buffer in self._buffers[:len(self._memory)]:
                glBindBuffer(self._target, buffer)
                glUnmapBuffer(self._target)
            glBindBuffer(self._target, 0)
        self._memory = []
        if self._buffers:
            glDeleteBuffers(len(self._buffers), self._buffers)
        self._buffers = []
        self._fences = []
        self.size = 0

    def memory(self, slot: int) -> np.ndarray:
        """A mapped slot's bytes, for the caller to view in its layout."""
        return self._memory[slot]

    def bind(self, slot: int) -> None:
        """The slot bound on its target, for the transfer that follows."""
        glBindBuffer(self._target, self._buffers[slot])

    def unbind(self) -> None:
        glBindBuffer(self._target, 0)

    def read(self, slot: int, out: np.ndarray) -> None:
        """An unmapped slot's first `out.nbytes` bytes copied into `out`."""
        self.bind(slot)
        glGetBufferSubData(self._target, 0, out.nbytes, out.ctypes.data_as(ctypes.c_void_p))
        self.unbind()

    def fence(self, slot: int) -> None:
        """A fence after the transfer just queued into `slot`, in place of the one it had."""
        self._drop(slot)
        self._fences[slot] = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0)

    def waiting(self, slot: int) -> bool:
        """Whether a transfer into `slot` has a fence still to pass or to be taken by `passed`."""
        return self._fences[slot] is not None

    def passed(self, slot: int) -> bool:
        """Whether the slot's fence has passed, without waiting: True once, its fence then dropped."""
        fence = self._fences[slot]
        if fence is None or glClientWaitSync(fence, 0, 0) not in (GL_ALREADY_SIGNALED, GL_CONDITION_SATISFIED):
            return False
        self._drop(slot)
        return True

    def reset(self) -> None:
        """Every fence dropped: nothing waits."""
        for slot in range(len(self._fences)):
            self._drop(slot)

    def _drop(self, slot: int) -> None:
        fence = self._fences[slot]
        if fence is not None:
            glDeleteSync(fence)
            self._fences[slot] = None
