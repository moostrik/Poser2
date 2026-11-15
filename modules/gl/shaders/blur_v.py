from OpenGL.GL import GL_TEXTURE0, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1f, glUniform1i, glUniform2f, glUseProgram
from .. import Shader, draw_quad, Texture

import logging
logger = logging.getLogger(__name__)

class BlurV(Shader):
    def use(self, src: Texture, radius: float) -> None:
        if not self.allocated or not self.shader_program:
            logger.warning("BlurV shader not allocated or shader program missing.")
            return
        if not src.allocated:
            logger.warning("BlurV shader: input texture not allocated.")
            return

        # Activate shader program
        glUseProgram(self.shader_program)

        # Bind input texture
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, src.tex_id)

        # Configure shader uniforms (using cached locations)
        glUniform1i(self.get_uniform_loc("src"), 0)
        glUniform1f(self.get_uniform_loc("radius"), radius)
        glUniform2f(self.get_uniform_loc("resolution"), src.width, src.height)

        # Render
        draw_quad()

