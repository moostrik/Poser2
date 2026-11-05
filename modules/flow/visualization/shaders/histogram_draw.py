"""Histogram Draw shader.

Draws the bins `HistogramCount` filled: four lanes, or one channel's lane.
"""

from OpenGL.GL import GL_TEXTURE0, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1f, glUniform1i, glUseProgram
from modules.gl import Shader, draw_quad, Texture


class HistogramDraw(Shader):
    """Draws a histogram, its height the log of the count."""

    def use(self, histogram: Texture, channel: int, total: int) -> None:
        """Draw the bins.

        Args:
            histogram: R32UI bins, one row per channel
            channel: 0 to 3 for that channel's lane alone, 4 for all four
            total: Samples counted, the count drawn at full height
        """
        if not self.allocated or not self.shader_program:
            return
        if not histogram.allocated:
            return

        glUseProgram(self.shader_program)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, histogram.tex_id)

        glUniform1i(self.get_uniform_loc("tex0"), 0)
        glUniform1i(self.get_uniform_loc("channel"), channel)
        glUniform1f(self.get_uniform_loc("total"), float(total))

        draw_quad()
