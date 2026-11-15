from OpenGL.GL import GL_TEXTURE0, GL_TEXTURE1, GL_TEXTURE2, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1f, glUniform1i, glUseProgram
from .. import Shader, draw_quad, Texture

import logging
logger = logging.getLogger(__name__)

class NoiseBlend(Shader):
    def use(self, src: Texture, dst: Texture, mask: Texture, blend: float) -> None:
        if not self.allocated or not self.shader_program:
            logger.warning("NoiseBlend shader not allocated or shader program missing.")
            return
        if not src.allocated or not dst.allocated or not mask.allocated:
            logger.warning("NoiseBlend shader: input textures not allocated.")
            return

        # Activate shader program
        glUseProgram(self.shader_program)

        # Bind input textures
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, src.tex_id)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, dst.tex_id)
        glActiveTexture(GL_TEXTURE2)
        glBindTexture(GL_TEXTURE_2D, mask.tex_id)

        # Configure shader uniforms (using cached locations)
        glUniform1i(self.get_uniform_loc("src"), 0)
        glUniform1i(self.get_uniform_loc("dst"), 1)
        glUniform1i(self.get_uniform_loc("mask"), 2)
        glUniform1f(self.get_uniform_loc("blend"), blend)

        # Render
        draw_quad()

