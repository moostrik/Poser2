"""Time Series shader.

Draws one lane per series from the per-column lowest and highest samples.
"""

from OpenGL.GL import GL_TEXTURE0, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1f, glUniform1i, glUniform4fv, glUseProgram
from modules.gl import Shader, draw_quad, Texture


class TimeSeriesShader(Shader):
    """Draws lanes of time series."""

    def use(self, ranges: Texture, num_lanes: int, num_columns: int, line_width: float,
            colors: list[tuple[float, float, float, float]]) -> None:
        """Draw the lanes.

        Args:
            ranges: RG lowest and highest per column and lane, normalized; negative where a column has no sample
            num_lanes: Number of series
            num_columns: Columns per lane
            line_width: Line thickness in lane heights
            colors: Up to 8 RGBA colors, one per lane in turn
        """
        if not self.allocated or not self.shader_program:
            return
        if not ranges.allocated:
            return

        glUseProgram(self.shader_program)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, ranges.tex_id)

        padded: list[float] = [component for color in (colors * 8)[:8] for component in color]
        glUniform1i(self.get_uniform_loc("tex0"), 0)
        glUniform1i(self.get_uniform_loc("num_lanes"), num_lanes)
        glUniform1i(self.get_uniform_loc("num_columns"), num_columns)
        glUniform1f(self.get_uniform_loc("line_width"), line_width)
        glUniform4fv(self.get_uniform_loc("colors"), 8, padded)

        draw_quad()
