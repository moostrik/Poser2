"""Normalize shader.

Normalizes vectors to unit length.
"""

from OpenGL.GL import glUseProgram
from modules.gl import Shader, draw_quad, Texture

import logging
logger = logging.getLogger(__name__)


class Normalize(Shader):
    """Normalize vectors to unit length."""

    def use(self, source: Texture) -> None:
        """Normalize vectors in source texture.

        Args:
            source: Source texture
        """
        if not self.allocated or not self.shader_program:
            logger.warning("Normalize shader not allocated or shader program missing.")
            return
        if not source.allocated:
            logger.warning("Normalize shader: input texture not allocated.")
            return

        glUseProgram(self.shader_program)
        source.bind(0)
        draw_quad()
