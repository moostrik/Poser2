"""Kinetic Projection shader.

Projects the difference of the fast and slow trails on the direction of the motion.
"""

from enum import IntEnum

from OpenGL.GL import GL_TEXTURE0, GL_TEXTURE1, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1i, glUseProgram
from modules.gl import Shader, draw_quad, Texture


class Projection(IntEnum):
    ACCELERATION =       0     # along the motion
    TURN =               1     # across it
    DEPTH_ACCELERATION = 2     # the z of fast - slow


class KineticProjection(Shader):
    """Writes fast - slow projected as asked to the red channel."""

    def use(self, fast: Texture, slow: Texture, mode: Projection) -> None:
        """Project the trail difference.

        Args:
            fast: RGB fast trail
            slow: RGB slow trail
            mode: The projection
        """
        if not self.allocated or not self.shader_program:
            return
        if not fast.allocated or not slow.allocated:
            return

        glUseProgram(self.shader_program)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, fast.tex_id)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, slow.tex_id)

        glUniform1i(self.get_uniform_loc("tex0"), 0)
        glUniform1i(self.get_uniform_loc("tex1"), 1)
        glUniform1i(self.get_uniform_loc("mode"), int(mode))

        draw_quad()
