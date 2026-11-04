"""Depth Guide shader.

The depth sharpened at the image's edges: a joint bilateral filter guided by the luminance.
"""

from OpenGL.GL import GL_TEXTURE0, GL_TEXTURE1, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1f, glUniform1i, glUseProgram
from modules.gl import Shader, draw_quad, Texture


class DepthGuide(Shader):
    """Writes the depth (R, m) as the luminance-weighted mean of its neighborhood, 0 where it has none."""

    def use(self, depth: Texture, luminance: Texture, spacing: float) -> None:
        """Sharpen the depth at the image's edges.

        Args:
            depth: Depth (m), 0 where there is none
            luminance: The guide (0 to 1), any size
            spacing: Texels between two taps
        """
        if not self.allocated or not self.shader_program:
            return
        if not depth.allocated or not luminance.allocated:
            return

        glUseProgram(self.shader_program)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, depth.tex_id)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, luminance.tex_id)

        glUniform1i(self.get_uniform_loc("tex0"), 0)
        glUniform1i(self.get_uniform_loc("tex1"), 1)
        glUniform1f(self.get_uniform_loc("spacing"), spacing)

        draw_quad()
