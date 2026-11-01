"""Speed 3D shader.

The length of a 2D velocity and its velocity toward the camera together.
"""

from OpenGL.GL import GL_TEXTURE0, GL_TEXTURE1, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1i, glUseProgram
from modules.gl import Shader, draw_quad, Texture


class Speed3D(Shader):
    """Writes the speed of (x, y, z) to the red channel."""

    def use(self, velocity: Texture, depth_velocity: Texture) -> None:
        """Compute the speed.

        Args:
            velocity: RG velocity
            depth_velocity: Velocity toward the camera in the red channel
        """
        if not self.allocated or not self.shader_program:
            return
        if not velocity.allocated or not depth_velocity.allocated:
            return

        glUseProgram(self.shader_program)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, velocity.tex_id)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, depth_velocity.tex_id)

        glUniform1i(self.get_uniform_loc("tex0"), 0)
        glUniform1i(self.get_uniform_loc("tex1"), 1)

        draw_quad()
