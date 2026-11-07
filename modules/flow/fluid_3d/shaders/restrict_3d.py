"""Restrict3D - a volume onto the coarser level, for the multigrid."""
from __future__ import annotations

from OpenGL.GL import GL_R32F, glUniform3i, glUseProgram

from modules.gl import Texture3D, ComputeShader


class Restrict3D(ComputeShader):
    """Each coarse cell the mean of the fine cells it covers."""

    WORKGROUP_SIZE_X = 8
    WORKGROUP_SIZE_Y = 8
    WORKGROUP_SIZE_Z = 4

    def use(self, fine: Texture3D, coarse: Texture3D) -> None:
        if not self.allocated or not self.shader_program:
            return
        glUseProgram(self.shader_program)
        self.bind_image_3d_read(0, fine, GL_R32F)
        self.bind_image_3d_write(1, coarse, GL_R32F)
        glUniform3i(self.get_uniform_loc("uFineSize"), fine.width, fine.height, fine.depth)
        glUniform3i(self.get_uniform_loc("uCoarseSize"), coarse.width, coarse.height, coarse.depth)
        self.dispatch(coarse.width, coarse.height, coarse.depth)
