from OpenGL.GL import glUniform1f, glUniform2f, glUseProgram
from modules.gl import Shader, draw_quad, Texture

import logging
logger = logging.getLogger(__name__)

class Omission(Shader):
    def use(self, tex0: Texture, tex1: Texture, mask: Texture, bloom: float, omission: float) -> None:
        if not self.allocated or not self.shader_program:
            logger.warning("Omission shader not allocated or shader program missing.")
            return
        if not tex0.allocated or not tex1.allocated or not mask.allocated:
            logger.warning("Omission shader: input textures not allocated.")
            return

        # Activate shader program
        glUseProgram(self.shader_program)

        # Bind input textures
        tex0.bind(0)
        tex1.bind(1)
        mask.bind(2)

        # Configure shader uniforms (using cached locations)
        glUniform1f(self.get_uniform_loc("bloom"), bloom)
        glUniform1f(self.get_uniform_loc("omission"), omission)
        glUniform2f(self.get_uniform_loc("resolution"), tex0.width, tex0.height)

        # Render
        draw_quad()

