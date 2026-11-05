"""Strip Summary shader.

A texture's channels as amounts, in one small texture whose top mip is their mean.
"""

from OpenGL.GL import GL_TEXTURE0, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1i, glUseProgram
from modules.gl import Shader, draw_quad, Texture


class StripSummary(Shader):
    """Writes |x|, |y|, |z|, |w| of a texture to RGBA."""

    def use(self, texture: Texture) -> None:
        if not self.allocated or not self.shader_program:
            return
        if not texture.allocated:
            return

        glUseProgram(self.shader_program)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, texture.tex_id)

        glUniform1i(self.get_uniform_loc("tex0"), 0)

        draw_quad()
