"""Blit shader for copying/resizing textures.

Simple passthrough shader that copies texture data without modification.
"""

from OpenGL.GL import *  # type: ignore
from modules.gl.Shader import Shader, draw_quad
from modules.gl import Texture


class Blit(Shader):
    """Copy/Blit texture to target FBO."""

    def use(self, tex: Texture) -> None:
        """Render texture to current FBO.

        Args:
            tex: Source texture to copy
        """
        if not self.allocated or not self.shader_program:
            print("Blit shader not allocated or shader program missing.")
            return
        if not tex.allocated:
            return

        # Activate shader program
        glUseProgram(self.shader_program)

        # Bind input texture
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, tex.tex_id)

        # Configure shader uniform (using cached location)
        glUniform1i(self.get_uniform_loc("tex"), 0)

        # Render
        draw_quad()
