"""Prolong2D - the coarser level's correction added to a level, for the 2D multigrid."""
from __future__ import annotations

from OpenGL.GL import GL_R32F, glUniform1i, glUniform2i, glUseProgram

from modules.gl import Texture, ComputeShader


class Prolong2D(ComputeShader):
    """fine += the coarse correction, bilinear at the fine cell's center, beyond the field the walls' ghost."""

    WORKGROUP_SIZE_X = 16
    WORKGROUP_SIZE_Y = 16
    WORKGROUP_SIZE_Z = 1

    def use(self, coarse: Texture, fine: Texture, closed: bool = True) -> None:
        if not self.allocated or not self.shader_program:
            return
        glUseProgram(self.shader_program)
        self.bind_image_read(1, coarse, GL_R32F)
        self.bind_image_readwrite(0, fine, GL_R32F)
        glUniform2i(self.get_uniform_loc("uFineSize"), fine.width, fine.height)
        glUniform2i(self.get_uniform_loc("uCoarseSize"), coarse.width, coarse.height)
        glUniform1i(self.get_uniform_loc("uClosed"), int(closed))
        self.dispatch(fine.width, fine.height)
