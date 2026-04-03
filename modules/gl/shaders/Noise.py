from OpenGL.GL import * # type: ignore
from modules.gl.Shader import Shader, draw_quad
 # ...existing code...
from random import random

import logging
logger = logging.getLogger(__name__)

class Noise(Shader):
    def use(self) -> None:
        if not self.allocated or not self.shader_program:
            logger.warning("Noise shader not allocated or shader program missing.")
            return

        # Activate shader program
        glUseProgram(self.shader_program)

        # Configure shader uniforms (using cached locations)
        glUniform1f(self.get_uniform_loc("random"), random())

        # Render
        draw_quad()

