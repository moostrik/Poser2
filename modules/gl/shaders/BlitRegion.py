"""Shader for drawing a region of a texture to fill the viewport.

Renders a specific rectangular region (x, y, w, h) of a texture,
stretched to fill the entire current viewport.
"""

from OpenGL.GL import *  # type: ignore
from modules.gl.Shader import Shader, draw_quad
from modules.gl import Texture


class BlitRegion(Shader):
    """Draw a region of a texture to fill the entire viewport."""

    def use(self, tex: Texture, x: float, y: float, w: float, h: float) -> None:
        """Render a region of the texture to fill the current FBO.

        Args:
            tex: Source texture to draw from
            x: X position in normalized texture coordinates [0, 1]
            y: Y position in normalized texture coordinates [0, 1]
            w: Width in normalized texture coordinates [0, 1]
            h: Height in normalized texture coordinates [0, 1]
        """
        if not self.allocated or not self.shader_program:
            print("BlitRegion shader not allocated or shader program missing.")
            return
        if not tex.allocated:
            print("BlitRegion shader: input texture not allocated.")
            return

        # Activate shader program
        glUseProgram(self.shader_program)

        # Bind input texture
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, tex.tex_id)

        # Configure shader uniforms
        glUniform1i(self.get_uniform_loc("tex"), 0)
        glUniform4f(self.get_uniform_loc("region"), x, y, w, h)

        # Render
        draw_quad()

        # Cleanup
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, 0)
        glUseProgram(0)
