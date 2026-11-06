"""Velocity Event Region shader.

Writes the flow's velocity where a pixel moves and whether it moves, for the start's region means.
"""

from OpenGL.GL import GL_TEXTURE0, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1f, glUniform1i, glUseProgram
from modules.gl import Shader, draw_quad, Texture


class VelocityEventRegion(Shader):
    """Writes (velocity * moving, moving) to RGBA."""

    def use(self, velocity: Texture, floor_speed: float) -> None:
        """Write the moving velocity.

        Args:
            velocity: RGB velocity, image widths per second
            floor_speed: A pixel slower than this does not move
        """
        if not self.allocated or not self.shader_program:
            return
        if not velocity.allocated:
            return

        glUseProgram(self.shader_program)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, velocity.tex_id)

        glUniform1i(self.get_uniform_loc("tex0"), 0)
        glUniform1f(self.get_uniform_loc("floor_speed"), floor_speed)

        draw_quad()
