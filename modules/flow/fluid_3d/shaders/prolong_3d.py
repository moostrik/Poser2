"""Prolong3D - the coarser level's correction added to a level, for the multigrid."""
from __future__ import annotations

from OpenGL.GL import GL_R32F, glUniform3i, glUseProgram

from modules.gl import Texture3D, ComputeShader


class Prolong3D(ComputeShader):
    """fine += the coarse correction, trilinear at the fine cell's center."""

    WORKGROUP_SIZE_X = 8
    WORKGROUP_SIZE_Y = 8
    WORKGROUP_SIZE_Z = 4

    def use(self, coarse: Texture3D, fine: Texture3D) -> None:
        if not self.allocated or not self.shader_program:
            return
        glUseProgram(self.shader_program)
        self.bind_texture_3d(0, coarse, "uCoarse")
        self.bind_image_3d_readwrite(0, fine, GL_R32F)
        glUniform3i(self.get_uniform_loc("uFineSize"), fine.width, fine.height, fine.depth)
        self.dispatch(fine.width, fine.height, fine.depth)
