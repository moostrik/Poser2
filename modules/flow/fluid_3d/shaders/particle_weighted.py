"""ParticleWeighted - the weighted look's targets into the 2D output."""
from __future__ import annotations

from OpenGL.GL import GL_RGBA16F, glUniform2i, glUseProgram

from modules.gl import ComputeShader, Texture


class ParticleWeighted(ComputeShader):
    """The weighted emission over the weighted opacity, times the opacity the revealage leaves."""

    WORKGROUP_SIZE_X = 16
    WORKGROUP_SIZE_Y = 16
    WORKGROUP_SIZE_Z = 1

    def use(self, weighted_sum: Texture, reveal: Texture, output: Texture) -> None:
        if not self.allocated or not self.shader_program:
            return
        glUseProgram(self.shader_program)
        self.bind_texture(0, weighted_sum, "uWeightedSum")
        self.bind_texture(1, reveal, "uReveal")
        self.bind_image_write(0, output, GL_RGBA16F)
        glUniform2i(self.get_uniform_loc("uOutputSize"), output.width, output.height)
        self.dispatch(output.width, output.height)
