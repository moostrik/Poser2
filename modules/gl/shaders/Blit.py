"""Blit shader for copying/resizing textures.

Simple passthrough shader that copies texture data without modification.
"""

from OpenGL.GL import *  # type: ignore
from modules.gl.Shader import Shader, draw_quad
from modules.gl import Texture

import logging
logger = logging.getLogger(__name__)


class Blit(Shader):
    """Copy/stretch texture to target FBO."""

    def use(self, tex: Texture, opacity: float = 1.0) -> None:
        """Render texture to current FBO.

        Args:
            tex: Source texture to copy
            opacity: Output alpha multiplier (0.0–1.0)
        """
        if not self.allocated or not self.shader_program:
            logger.warning("Blit shader not allocated or shader program missing.")
            return
        if not tex.allocated:
            return

        # Activate shader program
        glUseProgram(self.shader_program)

        # Bind input texture
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, tex.tex_id)

        # Configure shader uniforms (using cached locations)
        glUniform1i(self.get_uniform_loc("tex"), 0)
        glUniform1f(self.get_uniform_loc("opacity"), opacity)

        # Render
        draw_quad()
