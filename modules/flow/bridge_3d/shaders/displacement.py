"""Displacement shader: the rate at which a moving body fills the water, from its velocity and its mask."""

from OpenGL.GL import GL_TEXTURE0, GL_TEXTURE1, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1i, glUseProgram
from modules.gl import Shader, draw_quad, Texture


class Displacement(Shader):
    """Writes −v·∇mask (1/s) to red and the velocity's z times the mask (widths/s) to green."""

    def use(self, velocity: Texture, mask: Texture) -> None:
        """The displacement of the body at `velocity` (xyz, widths/s) within `mask` (red, 0 to 1)."""
        if not self.allocated or not self.shader_program or not velocity.allocated or not mask.allocated:
            return

        glUseProgram(self.shader_program)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, velocity.tex_id)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, mask.tex_id)

        glUniform1i(self.get_uniform_loc("velocity"), 0)
        glUniform1i(self.get_uniform_loc("mask"), 1)

        draw_quad()
