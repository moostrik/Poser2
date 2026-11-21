from OpenGL.GL import glUniform1f, glUseProgram
from ... import Shader, draw_quad, Texture

import logging
logger = logging.getLogger(__name__)

class Exposure(Shader):
    def use(self, src: Texture, exposure: float, offset: float, gamma: float) -> None:
        if not self.allocated or not self.shader_program:
            logger.warning("Exposure shader not allocated or shader program missing.")
            return
        if not src.allocated:
            logger.warning("Exposure shader: input texture not allocated.")
            return

        # Activate shader program
        glUseProgram(self.shader_program)

        # Bind input texture
        src.bind(0)

        # Configure shader uniforms (using cached locations)
        glUniform1f(self.get_uniform_loc("exposure"), exposure)
        glUniform1f(self.get_uniform_loc("offset"), offset)
        glUniform1f(self.get_uniform_loc("gamma"), gamma)

        # Render
        draw_quad()

