"""Larger Amount shader: per pixel, the one of two textures with the larger amount."""

from OpenGL.GL import GL_TEXTURE0, GL_TEXTURE1, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1i, glUseProgram
from modules.gl import Shader, draw_quad, Texture


class LargerAmount(Shader):
    """Writes, per pixel, `held` or `source`, whichever has the larger amount, as it is."""

    def use(self, held: Texture, source: Texture, by_alpha: bool) -> None:
        """`by_alpha`: the amount is the alpha (a density, RGBA with its color); else the size of red (a temperature, ±)."""
        if not self.allocated or not self.shader_program or not held.allocated or not source.allocated:
            return

        glUseProgram(self.shader_program)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, held.tex_id)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, source.tex_id)

        glUniform1i(self.get_uniform_loc("held"), 0)
        glUniform1i(self.get_uniform_loc("source"), 1)
        glUniform1i(self.get_uniform_loc("byAlpha"), int(by_alpha))

        draw_quad()
