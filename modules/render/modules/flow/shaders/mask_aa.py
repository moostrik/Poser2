from OpenGL.GL import GL_TEXTURE0, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1i, glUseProgram
from modules.gl import Shader, draw_quad, Texture

import logging
logger = logging.getLogger(__name__)


class MaskAA(Shader):
    """A mask's red thresholded at 0.5, its edge antialiased over one pixel."""

    def use(self, tex0: Texture) -> None:
        if not self.allocated or not self.shader_program:
            logger.warning("MaskAA shader not allocated or shader program missing.")
            return
        if not tex0.allocated:
            logger.warning("MaskAA shader: input texture not allocated.")
            return

        glUseProgram(self.shader_program)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, tex0.tex_id)
        glUniform1i(self.get_uniform_loc("tex0"), 0)
        draw_quad()
