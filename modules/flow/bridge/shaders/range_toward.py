"""Range toward shader: range flow's velocity toward the camera as the depth velocity."""

from OpenGL.GL import GL_TEXTURE0, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1f, glUniform1i, glUseProgram
from modules.gl import Shader, draw_quad, Texture


class RangeToward(Shader):
    """Writes the scaled velocity toward the camera to the red channel."""

    def use(self, motion: Texture, scale: float) -> None:
        """Take the velocity toward the camera from range flow's `motion` (B, m/s), times `scale`."""
        if not self.allocated or not self.shader_program or not motion.allocated:
            return

        glUseProgram(self.shader_program)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, motion.tex_id)

        glUniform1i(self.get_uniform_loc("motion"), 0)
        glUniform1f(self.get_uniform_loc("scale"), scale)

        draw_quad()
