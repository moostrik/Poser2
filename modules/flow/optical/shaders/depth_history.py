"""Depth History shader.

Keeps the depth of the last four frames at each pixel, for the depth track's restart.
"""

from OpenGL.GL import GL_TEXTURE0, GL_TEXTURE1, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1i, glUseProgram
from modules.gl import Shader, draw_quad, Texture


class DepthHistory(Shader):
    """Writes the depth now to red and the history the frame before one channel on (m)."""

    def use(self, depth: Texture, history: Texture) -> None:
        """Move the history one frame on.

        Args:
            depth: Depth now (m), at the history's size
            history: The history the frame before
        """
        if not self.allocated or not self.shader_program:
            return
        if not depth.allocated or not history.allocated:
            return

        glUseProgram(self.shader_program)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, depth.tex_id)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, history.tex_id)

        glUniform1i(self.get_uniform_loc("tex0"), 0)
        glUniform1i(self.get_uniform_loc("tex1"), 1)

        draw_quad()
