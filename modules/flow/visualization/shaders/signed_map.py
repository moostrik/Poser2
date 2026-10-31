"""Signed Map shader.

Draws a scalar field in two colors, one per sign.
"""

from OpenGL.GL import GL_TEXTURE0, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1f, glUniform1i, glUseProgram
from modules.gl import Shader, draw_quad, Texture


class SignedMap(Shader):
    """Maps a scalar field to a warm color when positive and a cold one when negative."""

    def use(self, field: Texture, scale: float) -> None:
        """Draw the field.

        Args:
            field: Scalar field in the red channel
            scale: Multiplier on the value
        """
        if not self.allocated or not self.shader_program:
            return
        if not field.allocated:
            return

        glUseProgram(self.shader_program)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, field.tex_id)

        glUniform1i(self.get_uniform_loc("tex0"), 0)
        glUniform1f(self.get_uniform_loc("scale"), scale)

        draw_quad()
