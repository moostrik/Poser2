"""The color a bridge's density carries: the image's RGB, or white without one, times a tint and the mask."""

from OpenGL.GL import GL_TEXTURE0, GL_TEXTURE1, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1i, glUniform3f, glUseProgram
from modules.gl import Shader, draw_quad, Texture

import logging
logger = logging.getLogger(__name__)


class DensityColor(Shader):
    """`image` times `color` times `mask`'s red, RGB and alpha alike; without an image the color alone,
    without a mask ungated."""

    def use(self, image: Texture | None, color: tuple[float, float, float], mask: Texture | None) -> None:
        if not self.allocated or not self.shader_program:
            logger.warning("DensityColor shader not allocated or shader program missing.")
            return
        has_image: bool = image is not None and image.allocated
        has_mask: bool = mask is not None and mask.allocated

        glUseProgram(self.shader_program)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, image.tex_id if has_image and image is not None else 0)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, mask.tex_id if has_mask and mask is not None else 0)

        glUniform1i(self.get_uniform_loc("image"), 0)
        glUniform1i(self.get_uniform_loc("mask"), 1)
        glUniform1i(self.get_uniform_loc("has_image"), int(has_image))
        glUniform1i(self.get_uniform_loc("has_mask"), int(has_mask))
        glUniform3f(self.get_uniform_loc("color"), *color)

        draw_quad()
