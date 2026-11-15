from OpenGL.GL import GL_TEXTURE0, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1f, glUniform1i, glUniform2f, glUseProgram
from .. import Shader, draw_quad, Texture

import logging
logger = logging.getLogger(__name__)

class MaskErode(Shader):
    def use(self, src: Texture, radius: float = 1.0, texel_size: tuple | None = None) -> None:
        if not self.allocated or not self.shader_program:
            logger.warning("MaskErode shader not allocated or shader program missing.")
            return
        if not src.allocated:
            logger.warning("MaskErode shader: input texture not allocated.")
            return

        if texel_size is None:
            texel_size = (1.0 / src.width, 1.0 / src.height)

        # Activate shader program
        glUseProgram(self.shader_program)

        # Bind input texture
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, src.tex_id)

        # Configure shader uniforms (using cached locations)
        glUniform1i(self.get_uniform_loc("src"), 0)
        glUniform1f(self.get_uniform_loc("radius"), radius)
        glUniform2f(self.get_uniform_loc("texelSize"), texel_size[0], texel_size[1])

        # Render
        draw_quad()
