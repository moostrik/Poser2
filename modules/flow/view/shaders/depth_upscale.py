"""Depth Upscale shader: the aligned depth read into the view across a surface (`view_window.edge_aware_depth`)."""

from OpenGL.GL import GL_TEXTURE0, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1f, glUniform1i, glUniform4f, glUseProgram
from modules.gl import Shader, draw_quad, Texture
from modules.utils import Rect


class DepthUpscale(Shader):
    """Writes the depth in the view's window, metres in red: interpolated within a surface, the nearest across an edge."""

    def use(self, depth: Texture, roi: Rect, tolerance: float) -> None:
        """`depth` metres (0 for none) on the camera image, `roi` the view's window on it with rows up, `tolerance`
        (m) the spread of four depth pixels below which they are one surface."""
        if not self.allocated or not self.shader_program or not depth.allocated:
            return

        glUseProgram(self.shader_program)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, depth.tex_id)

        glUniform1i(self.get_uniform_loc("depth"), 0)
        glUniform4f(self.get_uniform_loc("roi"), roi.x, roi.y, roi.width, roi.height)
        glUniform1f(self.get_uniform_loc("tolerance"), tolerance)

        draw_quad()
