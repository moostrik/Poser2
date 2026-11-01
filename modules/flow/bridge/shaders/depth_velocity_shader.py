"""Depth Velocity shader.

The velocity toward the camera, from two smoothed depths.
"""

from OpenGL.GL import GL_TEXTURE0, GL_TEXTURE1, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1f, glUniform1i, glUseProgram
from modules.gl import Shader, draw_quad, Texture


class DepthVelocityShader(Shader):
    """Writes the signed velocity toward the camera to the red channel."""

    def use(self, depth: Texture, previous: Texture, dt: float, jump: float, scale: float) -> None:
        """Difference two depths.

        Args:
            depth: Smoothed depth
            previous: Previous smoothed depth
            dt: Seconds between them
            jump: More change than this is an edge (m)
            scale: Converts m/s to simulation velocity
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
        glUniform1f(self.get_uniform_loc("dt"), dt)
        glUniform1f(self.get_uniform_loc("jump"), jump)
        glUniform1f(self.get_uniform_loc("scale"), scale)

        draw_quad()
