from OpenGL.GL import GL_TEXTURE0, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1i, glUniform4f, glUseProgram
from modules.utils import Color
from ... import Shader, draw_quad, Texture

import logging
logger = logging.getLogger(__name__)


class Tint(Shader):
    """Draw a texture multiplied by a tint color. Modern GL replacement for glColor4f()."""

    def use(self, src: Texture, color: Color) -> None:
        if not self.allocated or not self.shader_program:
            logger.warning("Tint shader not allocated or shader program missing.")
            return
        if not src.allocated:
            logger.warning("Tint shader: input texture not allocated.")
            return

        glUseProgram(self.shader_program)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, src.tex_id)

        glUniform1i(self.get_uniform_loc("src"), 0)
        glUniform4f(self.get_uniform_loc("tint"), *color)

        draw_quad()
