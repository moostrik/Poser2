"""MaskAntialias: a mask thresholded at 0.5, its edge antialiased over one pixel."""

from OpenGL.GL import GL_TEXTURE0, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUseProgram
from ... import Shader, draw_quad, Texture

import logging
logger = logging.getLogger(__name__)


class MaskAntialias(Shader):
    """`tex0`'s red thresholded at 0.5, smoothed over the pixel the threshold crosses (`fwidth`), written to RGB."""

    def use(self, tex0: Texture) -> None:
        if not self.allocated or not self.shader_program:
            logger.warning("MaskAntialias shader not allocated or shader program missing.")
            return
        if not tex0.allocated:
            logger.warning("MaskAntialias shader: input texture not allocated.")
            return

        glUseProgram(self.shader_program)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, tex0.tex_id)
        draw_quad()
