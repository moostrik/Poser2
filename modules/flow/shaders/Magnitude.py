"""Magnitude shader.

Computes vector magnitude (length) for each pixel.
"""

from OpenGL.GL import *  # type: ignore
from modules.gl.Shader import Shader, draw_quad
from modules.gl import Texture

import logging
logger = logging.getLogger(__name__)


class Magnitude(Shader):
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
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, src.tex_id)
        glUniform1i(self.get_uniform_loc("tex"), 0)
        draw_quad()
