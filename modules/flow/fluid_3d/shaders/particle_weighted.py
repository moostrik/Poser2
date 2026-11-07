"""ParticleWeighted - the weighted look's targets into the 2D output."""
from __future__ import annotations

from OpenGL.GL import GL_RGBA16F, glUniform1f, glUniform2i, glUseProgram

from modules.gl import ComputeShader, Texture


class ParticleWeighted(ComputeShader):
    """The weighted emission over the weighted opacity, times the opacity the revealage leaves, the glow mixed in."""

    WORKGROUP_SIZE_X = 16
    WORKGROUP_SIZE_Y = 16
    WORKGROUP_SIZE_Z = 1

    def use(self, weighted_sum: Texture, reveal: Texture, additive: Texture, output: Texture, glow: float) -> None:
        if not self.allocated or not self.shader_program:
            return
        glUseProgram(self.shader_program)
        self.bind_texture(0, weighted_sum, "uWeightedSum")
        self.bind_texture(1, reveal, "uReveal")
        self.bind_texture(2, additive, "uAdditive")
        self.bind_image_write(0, output, GL_RGBA16F)
        glUniform2i(self.get_uniform_loc("uOutputSize"), output.width, output.height)
        glUniform1f(self.get_uniform_loc("uGlow"), glow)
        self.dispatch(output.width, output.height)
