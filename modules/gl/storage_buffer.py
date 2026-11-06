"""StorageBuffer - a shader storage buffer (GL_SHADER_STORAGE_BUFFER) that lives on the GPU.

The CPU allocates, clears and binds it; compute and vertex shaders read and write it. Nothing in it
is read back unless `read_uint` asks for one value.
"""
import ctypes

import numpy as np
from OpenGL.GL import (GL_DYNAMIC_COPY, GL_R32UI, GL_RED_INTEGER, GL_SHADER_STORAGE_BUFFER, GL_UNSIGNED_INT,
                       glBindBuffer, glBindBufferBase, glBufferData, glClearBufferData, glDeleteBuffers, glGenBuffers,
                       glGetBufferSubData)


class StorageBuffer:
    """A GPU buffer of `size` bytes for shader storage blocks; render thread only."""

    def __init__(self) -> None:
        self.buffer_id: int = 0
        self.size: int = 0

    @property
    def allocated(self) -> bool:
        return self.buffer_id != 0

    def allocate(self, size: int) -> None:
        """Allocate `size` bytes, zeroed."""
        self.deallocate()
        self.buffer_id = int(glGenBuffers(1))
        self.size = size
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.buffer_id)
        glBufferData(GL_SHADER_STORAGE_BUFFER, size, None, GL_DYNAMIC_COPY)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
        self.clear()

    def deallocate(self) -> None:
        if self.buffer_id:
            glDeleteBuffers(1, [self.buffer_id])
        self.buffer_id = 0
        self.size = 0

    def clear(self) -> None:
        """Every byte 0."""
        if not self.allocated:
            return
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.buffer_id)
        glClearBufferData(GL_SHADER_STORAGE_BUFFER, GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, np.zeros(1, np.uint32))
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

    def bind_base(self, index: int) -> None:
        """Bind to the storage block binding point `index`."""
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, index, self.buffer_id)

    def read_uint(self, offset: int = 0) -> int:
        """One uint at `offset` bytes; it waits for the GPU, so for a check, not for every frame."""
        value = np.zeros(1, np.uint32)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.buffer_id)
        glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, offset, 4, value.ctypes.data_as(ctypes.c_void_p))
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
        return int(value[0])
