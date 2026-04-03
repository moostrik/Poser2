"""Blit shader with optional horizontal and vertical flip."""

from OpenGL.GL import *  # type: ignore
from modules.gl.Shader import Shader, draw_quad
from modules.gl import Texture

import logging
logger = logging.getLogger(__name__)


class BlitFlip(Shader):
    """Copy texture to target with optional X/Y flip."""

    def use(self, tex: Texture, flip_x: bool = False, flip_y: bool = False) -> None:
        """Render texture to current FBO with optional flipping.

        Args:
            tex: Source texture to copy
            flip_x: Flip horizontally if True
            flip_y: Flip vertically if True
        """
        if not self.allocated or not self.shader_program:
            logger.warning("BlitFlip shader not allocated or shader program missing.")
            return
        if not tex.allocated:
            logger.warning("BlitFlip shader: input texture not allocated.")
            return

        glUseProgram(self.shader_program)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, tex.tex_id)

        glUniform1i(self.get_uniform_loc("tex"), 0)
        glUniform1i(self.get_uniform_loc("flipX"), flip_x)
        glUniform1i(self.get_uniform_loc("flipY"), flip_y)

        draw_quad()
