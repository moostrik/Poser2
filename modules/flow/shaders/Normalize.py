"""Normalize shader.

Normalizes vectors to unit length.
"""

from OpenGL.GL import *  # type: ignore
from modules.gl.Shader import Shader, draw_quad
from modules.gl import Texture


class Normalize(Shader):
    """Normalize vectors to unit length."""

    def use(self, source: Texture) -> None:
        """Normalize vectors in source texture.

        Args:
            source: Source texture
        """
        if not self.allocated or not self.shader_program:
            print("Normalize shader not allocated or shader program missing.")
            return
        if not source.allocated:
            print("Normalize shader: input texture not allocated.")
            return

        glUseProgram(self.shader_program)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, source.tex_id)
        glUniform1i(self.get_uniform_loc("tex"), 0)
        draw_quad()
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, 0)
        glUseProgram(0)
