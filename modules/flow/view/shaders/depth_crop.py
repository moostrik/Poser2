"""Depth Crop shader: the camera's depth inside the view's box on its own pixels (`depth_read.depth_crop`)."""

from OpenGL.GL import GL_TEXTURE0, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1f, glUniform1i, glUniform4f, glUseProgram
from modules.gl import Shader, draw_quad, Texture
from modules.utils import Rect


class DepthCrop(Shader):
    """Writes the depth inside the box, metres in red: the depth pixel each view pixel lies in where it is from
    `near` to `far`, else 0."""

    def use(self, depth: Texture, roi: Rect, near: float, far: float) -> None:
        """`depth` metres (0 for none) on the camera image, `roi` the box's sides on it (0..1, origin top-left),
        `near`, `far` (m) its front and back."""
        if not self.allocated or not self.shader_program or not depth.allocated:
            return

        glUseProgram(self.shader_program)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, depth.tex_id)
        glUniform1i(self.get_uniform_loc("depth"), 0)

        glUniform4f(self.get_uniform_loc("roi"), roi.x, roi.y, roi.width, roi.height)
        glUniform1f(self.get_uniform_loc("near"), near)
        glUniform1f(self.get_uniform_loc("far"), far)

        draw_quad()
