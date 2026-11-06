"""ReduceDensity - the density's 2D array averaged down into a 3D texture at the simulation's size."""
from __future__ import annotations

from OpenGL.GL import GL_RGBA16F, glUniform3i, glUseProgram

from modules.gl import ComputeShader, Texture2DArray, Texture3D


class ReduceDensity(ComputeShader):
    """Each texel of `target` the mean of the block of `source` texels it covers, layer for layer."""

    WORKGROUP_SIZE_X = 16
    WORKGROUP_SIZE_Y = 16
    WORKGROUP_SIZE_Z = 1

    def use(self, source: Texture2DArray, target: Texture3D) -> None:
        """Reduce `source` into `target`, which has as many layers.

        Args:
            source: The density (RGBA16F 2D array), at the output size
            target: The reduced density (RGBA16F 3D texture), at the simulation's width and height
        """
        if not self.allocated or not self.shader_program:
            return
        if not source.allocated or not target.allocated:
            return
        glUseProgram(self.shader_program)
        self.bind_texture_2d_array(0, source, "uSource")
        self.bind_image_3d_write(0, target, GL_RGBA16F)
        glUniform3i(self.get_uniform_loc("uSize"), target.width, target.height, target.depth)
        self.dispatch(target.width, target.height, target.depth)
