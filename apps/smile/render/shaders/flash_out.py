from OpenGL.GL import glUniform1f, glUseProgram
from modules.gl import Shader, draw_quad, Texture

import logging
logger = logging.getLogger(__name__)

class FlashOut(Shader):
    def use(self, tex0: Texture, tex1: Texture, tex2: Texture, flash: float) -> None:
        if not self.allocated or not self.shader_program:
            logger.warning("FlashOut shader not allocated or shader program missing.")
            return
        if not tex0.allocated or not tex1.allocated or not tex2.allocated:
            logger.warning("FlashOut shader: input textures not allocated.")
            return

        # Activate shader program
        glUseProgram(self.shader_program)

        # Bind input textures
        tex0.bind(0)
        tex1.bind(1)
        tex2.bind(2)

        # Configure shader uniforms (using cached locations)
        glUniform1f(self.get_uniform_loc("flash"), flash)

        # Render
        draw_quad()

