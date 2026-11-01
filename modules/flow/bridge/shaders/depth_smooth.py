"""Depth Smooth shader.

Keeps the player's depth through an Ema, restarting a pixel that jumps.
"""

from OpenGL.GL import GL_TEXTURE0, GL_TEXTURE1, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1f, glUniform1i, glUseProgram
from modules.gl import Shader, draw_quad, Texture


class DepthSmooth(Shader):
    """Writes the smoothed depth in metres to the red channel, 0 off the player."""

    def use(self, depth: Texture, previous: Texture, weight: float, jump: float) -> None:
        """Advance the smoothed depth.

        Args:
            depth: Metres in red, mask in alpha
            previous: Previous smoothed depth
            weight: Weight of the previous value
            jump: More change than this per frame is an edge (m)
        """
        if not self.allocated or not self.shader_program:
            return
        if not depth.allocated or not previous.allocated:
            return

        glUseProgram(self.shader_program)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, depth.tex_id)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, previous.tex_id)

        glUniform1i(self.get_uniform_loc("tex0"), 0)
        glUniform1i(self.get_uniform_loc("tex1"), 1)
        glUniform1f(self.get_uniform_loc("weight"), weight)
        glUniform1f(self.get_uniform_loc("jump"), jump)

        draw_quad()
