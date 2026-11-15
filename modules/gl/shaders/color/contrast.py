from OpenGL.GL import GL_TEXTURE0, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1f, glUniform1i, glUseProgram
from ... import Shader, draw_quad, Texture

import logging
logger = logging.getLogger(__name__)

class Contrast(Shader):
    def use(self, src: Texture, brightness: float, contrast: float) -> None:
        if not self.allocated or not self.shader_program:
            logger.warning("Contrast shader not allocated or shader program missing.")
            return
        if not src.allocated:
            logger.warning("Contrast shader: input texture not allocated.")
            return

        # Activate shader program
        glUseProgram(self.shader_program)

        # Bind input texture
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, src.tex_id)

        # Configure shader uniforms (using cached locations)
        glUniform1i(self.get_uniform_loc("src"), 0)
        glUniform1f(self.get_uniform_loc("brightness"), brightness)
        glUniform1f(self.get_uniform_loc("contrast"), contrast)

        # Render
        draw_quad()

