from OpenGL.GL import GL_TEXTURE0, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1i, glUseProgram
from modules.gl import Shader, draw_quad, Texture

import logging
logger = logging.getLogger(__name__)


class BlitRotated(Shader):
    """Draw a texture turned 90 degrees counterclockwise: its top along the left edge."""

    def use(self, tex: Texture) -> None:
        if not self.allocated or not self.shader_program:
            logger.warning("BlitRotated shader not allocated or shader program missing.")
            return
        if not tex.allocated:
            return

        glUseProgram(self.shader_program)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, tex.tex_id)
        glUniform1i(self.get_uniform_loc("tex"), 0)

        draw_quad()
