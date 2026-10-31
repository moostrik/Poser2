"""Kinetic Projection shader.

Projects the difference of the fast and slow trails on the direction of the motion.
"""

from OpenGL.GL import GL_TEXTURE0, GL_TEXTURE1, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1i, glUseProgram
from modules.gl import Shader, draw_quad, Texture


class KineticProjection(Shader):
    """Writes fast - slow along the motion (acceleration) or across it (turn) to the red channel."""

    def use(self, fast: Texture, slow: Texture, turn: bool) -> None:
        """Project the trail difference.

        Args:
            fast: RG fast trail
            slow: RG slow trail
            turn: Output the turn instead of the acceleration
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
        glUniform1i(self.get_uniform_loc("turn"), 1 if turn else 0)

        draw_quad()
