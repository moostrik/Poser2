"""ParticleRing - the ring's head back into the ring's size, and the indirect draw and dispatch over its filled slots."""
from __future__ import annotations

from OpenGL.GL import GL_COMMAND_BARRIER_BIT, glMemoryBarrier, glUniform1ui, glUseProgram

from modules.gl import ComputeShader, StorageBuffer

RING_BYTES: int = 48        # head, filled, two pads, the draw command (16), the dispatch command (12), a pad
DRAW_OFFSET: int = 16       # the DrawArraysIndirectCommand's byte offset in the ring
DISPATCH_OFFSET: int = 32   # the DispatchIndirectCommand's


class ParticleRing(ComputeShader):
    """Wrap the ring's head into [0, count); write the draw and the advection's dispatch over the filled slots."""

    WORKGROUP_SIZE_X = 1
    WORKGROUP_SIZE_Y = 1
    WORKGROUP_SIZE_Z = 1

    def __init__(self) -> None:
        super().__init__()

    def use(self, ring: StorageBuffer, count: int, vertices: int, group_size: int) -> None:
        if not self.allocated or not self.shader_program:
            return
        glUseProgram(self.shader_program)
        ring.bind_base(1)
        glUniform1ui(self.get_uniform_loc("uCount"), count)
        glUniform1ui(self.get_uniform_loc("uVertices"), vertices)
        glUniform1ui(self.get_uniform_loc("uGroupSize"), group_size)
        self.dispatch(1, 1)
        self.barrier_buffer()
        glMemoryBarrier(GL_COMMAND_BARRIER_BIT)
