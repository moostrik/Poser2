from OpenGL.GL import * # type: ignore
from ... import Shader, draw_quad
from ... import Texture

import logging
logger = logging.getLogger(__name__)

class BlurV(Shader):
    def use(self, tex0: Texture, radius: float) -> None:
        if not self.allocated or not self.shader_program:
            logger.warning("BlurV shader not allocated or shader program missing.")
            return
        if not tex0.allocated:
            logger.warning("BlurV shader: input texture not allocated.")
            return

        # Activate shader program
        glUseProgram(self.shader_program)

        # Bind input texture
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, tex0.tex_id)

        # Configure shader uniforms (using cached locations)
        glUniform1i(self.get_uniform_loc("tex0"), 0)
        glUniform1f(self.get_uniform_loc("radius"), radius)

        # Render
        draw_quad()

