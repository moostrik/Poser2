"""Depth Threshold shader.

Takes the noise floor off the motion toward the camera.
"""

from OpenGL.GL import GL_TEXTURE0, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1f, glUniform1i, glUseProgram
from modules.gl import Shader, draw_quad, Texture


class DepthThreshold(Shader):
    """Writes the field with `threshold` taken off its size to the red channel."""

    def use(self, field: Texture, threshold: float) -> None:
        """Take the noise floor off.

        Args:
            field: Signed field in the red channel
            threshold: Size taken off, in the field's units
        """
        if not self.allocated or not self.shader_program:
            return
        if not field.allocated:
            return

        glUseProgram(self.shader_program)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, field.tex_id)

        glUniform1i(self.get_uniform_loc("tex0"), 0)
        glUniform1f(self.get_uniform_loc("threshold"), threshold)

        draw_quad()
