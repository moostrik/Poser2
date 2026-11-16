"""Depth Upscale shader: the aligned depth read into the view where it is the player's and the mask is
(`depth_read.upscaled_depth`)."""

from OpenGL.GL import GL_TEXTURE0, GL_TEXTURE1, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1f, glUniform1i, glUniform4f, glUseProgram
from modules.gl import Shader, draw_quad, Texture
from modules.utils import Rect


class DepthUpscale(Shader):
    """Writes the depth in the view's window, metres in red, where it is the player's and the mask is above 0, else 0."""

    def use(self, depth: Texture, roi: Rect, tolerance: float, mask: Texture, near: float, far: float) -> None:
        """`depth` metres (0 for none) on the camera image, `roi` the view's window on it (0..1, origin top-left), `tolerance`
        (m) the spread of four depth pixels below which they are one surface; `mask` the player's in the view, `near`,
        `far` (m) the player's depth range."""
        if not self.allocated or not self.shader_program or not depth.allocated or not mask.allocated:
            return

        glUseProgram(self.shader_program)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, depth.tex_id)
        glUniform1i(self.get_uniform_loc("depth"), 0)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, mask.tex_id)
        glUniform1i(self.get_uniform_loc("mask"), 1)

        glUniform4f(self.get_uniform_loc("roi"), roi.x, roi.y, roi.width, roi.height)
        glUniform1f(self.get_uniform_loc("tolerance"), tolerance)
        glUniform1f(self.get_uniform_loc("near"), near)
        glUniform1f(self.get_uniform_loc("far"), far)

        draw_quad()
