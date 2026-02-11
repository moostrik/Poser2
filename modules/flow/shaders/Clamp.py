"""Clamp shader.

Clamps all texture channels to a specified min/max range.
"""

from OpenGL.GL import *  # type: ignore
from modules.gl.Shader import Shader, draw_quad
from modules.gl import Texture


class Clamp(Shader):
    """Clamp texture values to a range."""

    def use(self, src: Texture, min_val: float, max_val: float) -> None:
        """Clamp source texture values to range.

        Args:
            src: Source texture
            min_val: Minimum value for all channels
            max_val: Maximum value for all channels
        """
        if not self.allocated or not self.shader_program:
            print("Clamp shader not allocated or shader program missing.")
            return
        if not src.allocated:
            print("Clamp shader: input texture not allocated.")
            return

        glUseProgram(self.shader_program)

        # Bind texture
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, src.tex_id)
        glUniform1i(self.get_uniform_loc("src"), 0)

        # Set uniforms
        glUniform1f(self.get_uniform_loc("minVal"), min_val)
        glUniform1f(self.get_uniform_loc("maxVal"), max_val)

        draw_quad()
