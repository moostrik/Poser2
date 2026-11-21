"""MaskAntialias: a mask thresholded at 0.5, its edge antialiased over one pixel."""

from OpenGL.GL import glUseProgram
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
        tex0.bind(0)
        draw_quad()
