from OpenGL.GL import glUniform1f, glUseProgram
from ... import Shader, draw_quad, Texture

import logging
logger = logging.getLogger(__name__)

class MaskApply(Shader):
    def use(self, src: Texture, mask: Texture, multiply: float = 1.0) -> None:
        if not self.allocated or not self.shader_program:
            logger.warning("MaskApply shader not allocated or shader program missing.")
            return
        if not src.allocated or not mask.allocated:
            logger.warning("MaskApply shader: input textures not allocated.")
            return

        # Activate shader program
        glUseProgram(self.shader_program)

        # Bind input textures
        src.bind(0)
        mask.bind(1)

        # Configure shader uniforms (using cached locations)
        glUniform1f(self.get_uniform_loc("multiply"), multiply)

        # Render
        draw_quad()

