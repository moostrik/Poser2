"""Velocity Event Region shader.

Writes the flow's velocity where a pixel moves and whether it moves, for a rise's region means.
"""

from OpenGL.GL import GL_TEXTURE0, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1i, glUseProgram
from modules.gl import Shader, draw_quad, Texture


class VelocityEventRegion(Shader):
    """Writes (velocity * moving, moving) to RGBA."""

    def use(self, velocity: Texture) -> None:
        """Write the moving velocity.

        Args:
            velocity: RGB velocity, image widths per second
        """
        if not self.allocated or not self.shader_program:
            return
        if not velocity.allocated:
            return

        glUseProgram(self.shader_program)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, velocity.tex_id)

        glUniform1i(self.get_uniform_loc("tex0"), 0)

        draw_quad()
