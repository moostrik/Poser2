"""Velocity Event Patch shader.

Writes the x/y velocities of the two trails, for the start gate's patch means.
"""

from OpenGL.GL import GL_TEXTURE0, GL_TEXTURE1, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1i, glUseProgram
from modules.gl import Shader, draw_quad, Texture


class VelocityEventPatch(Shader):
    """Writes (fast x, fast y, slow x, slow y) to RGBA."""

    def use(self, fast: Texture, slow: Texture) -> None:
        """Write the velocities.

        Args:
            fast: RGB fast trail
            slow: RGB slow trail
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

        draw_quad()
