"""Displacement median shader: the 5×5 median of a displacement, component by component."""

from OpenGL.GL import GL_TEXTURE0, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1i, glUniform2f, glUseProgram
from modules.gl import Shader, draw_quad, Texture


class DisplacementMedian(Shader):
    """Writes the displacement (RG, uv per frame) as the median of its 5×5 samples."""

    def use(self, displacement: Texture, spacing: tuple[float, float]) -> None:
        """Take the median.

        Args:
            displacement: uv per frame
            spacing: uv between two samples
        """
        if not self.allocated or not self.shader_program or not displacement.allocated:
            return

        glUseProgram(self.shader_program)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, displacement.tex_id)

        glUniform1i(self.get_uniform_loc("displacement"), 0)
        glUniform2f(self.get_uniform_loc("spacing"), *spacing)

        draw_quad()
