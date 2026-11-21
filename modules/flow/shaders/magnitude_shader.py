"""Magnitude shader.

Computes vector magnitude (length) for each pixel.
"""

from OpenGL.GL import glUseProgram
from modules.gl import Shader, draw_quad, Texture

import logging
logger = logging.getLogger(__name__)


class MagnitudeShader(Shader):
    """Compute vector magnitude from texture."""

    def use(self, src: Texture) -> None:
        """Compute magnitude of vector field.

        Args:
            source: Source texture
        """
        if not self.allocated or not self.shader_program:
            logger.warning("Magnitude shader not allocated or shader program missing.")
            return
        if not src.allocated:
            logger.warning("Magnitude shader: input texture not allocated.")
            return

        glUseProgram(self.shader_program)
        src.bind(0)
        draw_quad()
