"""Displacement threshold shader: the noise floor taken off a displacement."""

from OpenGL.GL import GL_TEXTURE0, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1f, glUniform1i, glUseProgram
from modules.gl import Shader, draw_quad, Texture


class DisplacementThreshold(Shader):
    """Writes the displacement (RG, uv per frame) with `threshold` pixels taken off its length."""

    def use(self, displacement: Texture, threshold: float) -> None:
        """Take the noise floor off.

        Args:
            displacement: uv per frame
            threshold: Taken off the length (px per frame)
        """
        if not self.allocated or not self.shader_program or not displacement.allocated:
            return

        glUseProgram(self.shader_program)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, displacement.tex_id)

        glUniform1i(self.get_uniform_loc("displacement"), 0)
        glUniform1f(self.get_uniform_loc("threshold"), threshold)

        draw_quad()
