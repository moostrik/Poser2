"""ParticleRing - the ring's head back into the ring's size."""
from __future__ import annotations

from OpenGL.GL import glUniform1ui, glUseProgram

from modules.gl import ComputeShader, StorageBuffer


class ParticleRing(ComputeShader):
    """Wrap the ring's head into [0, count)."""

    WORKGROUP_SIZE_X = 1
    WORKGROUP_SIZE_Y = 1
    WORKGROUP_SIZE_Z = 1

    def __init__(self) -> None:
        super().__init__()

    def use(self, ring: StorageBuffer, count: int) -> None:
        if not self.allocated or not self.shader_program:
            return
        glUseProgram(self.shader_program)
        ring.bind_base(1)
        glUniform1ui(self.get_uniform_loc("uCount"), count)
        self.dispatch(1, 1)
        self.barrier_buffer()
