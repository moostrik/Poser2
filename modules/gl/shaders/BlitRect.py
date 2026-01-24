"""Shader for drawing texture to a specific rectangle in viewport.

Renders a texture to a specified rectangular region (x, y, w, h) in the viewport,
with coordinates in normalized device space [-1, 1].
"""

from OpenGL.GL import *  # type: ignore
from modules.gl.Shader import Shader, draw_quad
from modules.gl import Texture


class BlitRect(Shader):
    """Draw texture to a specific rectangle in the viewport."""

    def use(self, tex: Texture, x: float, y: float, w: float, h: float) -> None:
        """Render texture to a rectangular region of the current FBO.

        Args:
            tex: Source texture to draw
            x: X position in normalized device coordinates [-1, 1]
            y: Y position in normalized device coordinates [-1, 1]
            w: Width in normalized device coordinates
            h: Height in normalized device coordinates
        """
        if not self.allocated or not self.shader_program:
            print("BlitRect shader not allocated or shader program missing.")
            return
        if not tex.allocated:
            print("BlitRect shader: input texture not allocated.")
            return

        # Activate shader program
        glUseProgram(self.shader_program)

        # Bind input texture
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, tex.tex_id)

        # Configure shader uniforms
        glUniform1i(self.get_uniform_loc("tex"), 0)
        glUniform4f(self.get_uniform_loc("rect"), x, y, w, h)

        # Render
        draw_quad()

        # Cleanup
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, 0)
        glUseProgram(0)
