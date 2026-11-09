"""Knee - the composited output's highlights bent softly toward 1 instead of clipped."""
from __future__ import annotations

from OpenGL.GL import GL_RGBA16F, glUniform2i, glUseProgram

from modules.gl import ComputeShader, Texture


class Knee(ComputeShader):
    """In place on the 2D output: each channel straight up to the knee, bent toward 1 above it."""

    WORKGROUP_SIZE_X = 16
    WORKGROUP_SIZE_Y = 16
    WORKGROUP_SIZE_Z = 1

    def use(self, output: Texture) -> None:
        if not self.allocated or not self.shader_program:
            return
        glUseProgram(self.shader_program)
        self.bind_image_readwrite(0, output, GL_RGBA16F)
        glUniform2i(self.get_uniform_loc("uOutputSize"), output.width, output.height)
        self.dispatch(output.width, output.height)
