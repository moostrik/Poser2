"""CompositeOver - one premultiplied image over the 2D output."""
from __future__ import annotations

from OpenGL.GL import GL_RGBA16F, glUniform2i, glUseProgram

from modules.gl import ComputeShader, Texture


class CompositeOver(ComputeShader):
    """The image over the output, Porter–Duff: the image's color plus the output × (1 - its alpha)."""

    WORKGROUP_SIZE_X = 16
    WORKGROUP_SIZE_Y = 16
    WORKGROUP_SIZE_Z = 1

    def use(self, over: Texture, output: Texture) -> None:
        if not self.allocated or not self.shader_program:
            return
        glUseProgram(self.shader_program)
        self.bind_texture(0, over, "uOver")
        self.bind_image_readwrite(0, output, GL_RGBA16F)
        glUniform2i(self.get_uniform_loc("uOutputSize"), output.width, output.height)
        self.dispatch(output.width, output.height)
