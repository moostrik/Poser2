"""Depth Upscale shader: the aligned depth read into the view (`view_window.edge_aware_depth`, `mask_sided_depth`,
`joint_bilateral_depth`), one of `DepthRead`'s ways."""

from OpenGL.GL import (GL_TEXTURE0, GL_TEXTURE1, GL_TEXTURE2, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1f,
                       glUniform1i, glUniform4f, glUseProgram)
from modules.gl import Shader, draw_quad, Texture
from modules.utils import Rect
from ..view_window import DepthRead


class DepthUpscale(Shader):
    """Writes the depth in the view's window, metres in red, the way `read` says."""

    def use(self, depth: Texture, roi: Rect, tolerance: float, read: DepthRead = DepthRead.SURFACE, mask: Texture | None = None,
            image: Texture | None = None, near: float = 0.0, far: float = 0.0, reach: float = 0.0, contrast: float = 0.1) -> None:
        """`depth` metres (0 for none) on the camera image, `roi` the view's window on it with rows up, `tolerance`
        (m) the spread of four depth pixels below which they are one surface. MASK: `mask` the player's in the view,
        `near`, `far` (m) the player's depth range and `reach` how far (depth pixels) a depth on the mask's side is
        looked for; BILATERAL: `image` the camera image and `contrast` the color distance (0 to 1) of the range
        gaussian's σ. Without what its way needs, SURFACE."""
        if not self.allocated or not self.shader_program or not depth.allocated:
            return
        if (read == DepthRead.MASK and (mask is None or not mask.allocated)) or \
                (read == DepthRead.BILATERAL and (image is None or not image.allocated)):
            read = DepthRead.SURFACE

        glUseProgram(self.shader_program)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, depth.tex_id)
        glUniform1i(self.get_uniform_loc("depth"), 0)
        if read == DepthRead.MASK and mask is not None:
            glActiveTexture(GL_TEXTURE1)
            glBindTexture(GL_TEXTURE_2D, mask.tex_id)
            glUniform1i(self.get_uniform_loc("mask"), 1)
        if read == DepthRead.BILATERAL and image is not None:
            glActiveTexture(GL_TEXTURE2)
            glBindTexture(GL_TEXTURE_2D, image.tex_id)
            glUniform1i(self.get_uniform_loc("image"), 2)

        glUniform4f(self.get_uniform_loc("roi"), roi.x, roi.y, roi.width, roi.height)
        glUniform1f(self.get_uniform_loc("tolerance"), tolerance)
        glUniform1i(self.get_uniform_loc("mode"), int(read))
        glUniform1f(self.get_uniform_loc("near"), near)
        glUniform1f(self.get_uniform_loc("far"), far)
        glUniform1f(self.get_uniform_loc("reach"), reach)
        glUniform1f(self.get_uniform_loc("contrast"), contrast)

        draw_quad()
