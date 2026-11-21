"""MaskMultiply: an image times a mask's red, every channel."""

from OpenGL.GL import GL_TEXTURE0, GL_TEXTURE1, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUseProgram
from ... import Shader, draw_quad, Texture

import logging
logger = logging.getLogger(__name__)


class MaskMultiply(Shader):
    """`tex0` times `tex1`'s red: the image premultiplied by the mask, beside `MaskApply`'s mask as alpha."""

    def use(self, tex0: Texture, tex1: Texture) -> None:
        if not self.allocated or not self.shader_program:
            logger.warning("MaskMultiply shader not allocated or shader program missing.")
            return
        if not tex0.allocated or not tex1.allocated:
            logger.warning("MaskMultiply shader: input textures not allocated.")
            return

        glUseProgram(self.shader_program)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, tex0.tex_id)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, tex1.tex_id)
        draw_quad()
