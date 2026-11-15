from OpenGL.GL import GL_TEXTURE0, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1f, glUniform1i, glUniform2f, glUseProgram
from .. import Shader, draw_quad, Texture

import logging
logger = logging.getLogger(__name__)

class MaskBlur(Shader):
    def use(self, src: Texture, horizontal: bool = True, radius: float = 1.0) -> None:
        if not self.allocated or not self.shader_program:
            logger.warning("MaskBlur shader not allocated or shader program missing.")
            return
        if not src.allocated:
            logger.warning("MaskBlur shader: input texture not allocated.")
            return

        # Set direction based on horizontal flag
        direction = (1.0, 0.0) if horizontal else (0.0, 1.0)

        # Activate shader program
        glUseProgram(self.shader_program)

        # Bind input texture
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, src.tex_id)

        # Configure shader uniforms (using cached locations)
        glUniform1i(self.get_uniform_loc("src"), 0)
        glUniform2f(self.get_uniform_loc("direction"), direction[0], direction[1])
        glUniform1f(self.get_uniform_loc("radius"), radius)

        # Render
        draw_quad()
