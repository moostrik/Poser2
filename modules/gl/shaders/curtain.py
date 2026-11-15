from OpenGL.GL import GL_TEXTURE0, GL_TEXTURE1, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1f, glUniform1i, glUseProgram
from .. import Shader, draw_quad, Texture

import logging
logger = logging.getLogger(__name__)

class Curtain(Shader):
    def use(self, src: Texture, dst: Texture, fade: float) -> None:
        if not self.allocated or not self.shader_program:
            logger.warning("Curtain shader not allocated or shader program missing.")
            return
        if not src.allocated or not dst.allocated:
            logger.warning("Curtain shader: input textures not allocated.")
            return

        # Activate shader program
        glUseProgram(self.shader_program)

        # Bind input textures
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, src.tex_id)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, dst.tex_id)

        # Configure shader uniforms (using cached locations)
        glUniform1i(self.get_uniform_loc("src"), 0)
        glUniform1i(self.get_uniform_loc("dst"), 1)
        glUniform1f(self.get_uniform_loc("fade"), fade)

        # Render
        draw_quad()

