"""Signed Map shader.

Draws a scalar field in two colors, one per sign.
"""

from OpenGL.GL import GL_TEXTURE0, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1f, glUniform1i, glUseProgram
from modules.gl import Shader, draw_quad, Texture


class SignedMap(Shader):
    """Maps a scalar field to a warm color when positive and a cold one when negative."""

    def use(self, field: Texture, scale: float, depth: bool = False, channel: int = 0) -> None:
        """Draw the field.

        Args:
            field: Scalar field in `channel`
            scale: Multiplier on the value
            depth: Red toward the camera and green away, instead of warm and cold
            channel: The field's channel, 0 for red
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
        glUniform1i(self.get_uniform_loc("depth"), 1 if depth else 0)
        glUniform1i(self.get_uniform_loc("channel"), channel)

        draw_quad()
