"""VelocityAverage3D - the velocity averaged in z over coarser layers."""
from __future__ import annotations

from OpenGL.GL import GL_RGBA16F, glUniform1i, glUniform3i, glUseProgram

from modules.gl import Texture3D, ComputeShader


class VelocityAverage3D(ComputeShader):
    """The velocity on the grid's x and y, averaged over the grid's layers each of the average's layers spans."""

    WORKGROUP_SIZE_X = 8
    WORKGROUP_SIZE_Y = 8
    WORKGROUP_SIZE_Z = 4

    def use(self, velocity: Texture3D, average: Texture3D) -> None:
        if not self.allocated or not self.shader_program:
            return
        glUseProgram(self.shader_program)
        self.bind_image_3d_read(0, velocity, GL_RGBA16F)
        self.bind_image_3d_write(1, average, GL_RGBA16F)
        glUniform3i(self.get_uniform_loc("uSize"), velocity.width, velocity.height, velocity.depth)
        glUniform1i(self.get_uniform_loc("uLayers"), average.depth)
        self.dispatch(average.width, average.height, average.depth)
