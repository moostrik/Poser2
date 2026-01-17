"""Magnitude shader.

Computes vector magnitude (length) for each pixel.
"""

from OpenGL.GL import *  # type: ignore
from modules.gl.Shader import Shader, draw_quad
from modules.gl import Texture


class Magnitude(Shader):
    """Compute vector magnitude from texture."""

    def use(self, source: Texture) -> None:
        """Compute magnitude of vector field.

        Args:
            source: Source texture
        """
        if not self.allocated or not self.shader_program:
            print("Magnitude shader not allocated or shader program missing.")
            return
        if not source.allocated:
            print("Magnitude shader: input texture not allocated.")
            return

        glUseProgram(self.shader_program)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, source.tex_id)
        glUniform1i(glGetUniformLocation(self.shader_program, "tex"), 0)
        draw_quad()
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, 0)
        glUseProgram(0)
