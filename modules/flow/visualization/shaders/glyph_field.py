"""Glyph Field shader.

Glyphs on a grid, procedural, without geometry shaders: arrows for a velocity, dots for a scalar.
Ported from ofxFlowTools visualization concepts.
"""

from OpenGL.GL import GL_TEXTURE0, GL_TEXTURE_2D, GL_VIEWPORT, glActiveTexture, glBindTexture, glGetIntegerv, glUniform1f, glUniform1i, glUniform2f, glUseProgram
from modules.gl import Shader, draw_quad, Texture


class GlyphField(Shader):
    """Samples a texture on a grid and draws a glyph per cell in the fragment shader.

    A velocity is an arrow: white for a 2D one; for a 3D one, the white arrow for x/y with a ring
    around the grid point for z, red toward the camera and green away. Sizes are on the cell, the grid
    spacing: at a speed of 1 the arrow reaches 1 cell and the ring's radius is 1/pi cell, the same
    length of line; faster, both roll off toward 2 cells. A scalar is a dot, in the colormap's color,
    its radius half a cell at 1, small values small.
    """

    def use(self, texture: Texture, scale: float = 1.0, spacing: float = 8.0, arrow_thickness: float = 0.8,
            xyz: bool = False, dots: bool = False, channel: int = 4) -> None:
        """Draw the glyphs into the target bound.

        Args:
            texture: The texture: a velocity (RG = x/y, B = z with xyz), or a scalar for dots
            scale: Multiplier on the values
            spacing: Distance between grid points in pixels, the cell the glyphs are sized on
            arrow_thickness: Arrow line thickness in pixels
            xyz: An arrow for a 3D velocity: the arrow with a ring for z
            dots: Dots for a scalar, not arrows
            channel: The dots' channel, 0 to 3, or 4 for all
        """
        if not self.allocated or not self.shader_program or not texture.allocated:
            return

        viewport = glGetIntegerv(GL_VIEWPORT)
        glUseProgram(self.shader_program)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, texture.tex_id)

        glUniform1i(self.get_uniform_loc("tex0"), 0)
        glUniform1i(self.get_uniform_loc("xyz"), 1 if xyz else 0)
        glUniform1i(self.get_uniform_loc("dots"), 1 if dots else 0)
        glUniform1i(self.get_uniform_loc("channel"), channel)
        glUniform1f(self.get_uniform_loc("scale"), scale)
        glUniform1f(self.get_uniform_loc("grid_spacing"), spacing)
        glUniform1f(self.get_uniform_loc("arrow_thickness"), arrow_thickness)
        glUniform2f(self.get_uniform_loc("resolution"), viewport[2], viewport[3])

        draw_quad()
