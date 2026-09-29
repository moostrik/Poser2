from OpenGL.GL import * # type: ignore
from .. import Shader, draw_quad
 # ...existing code...
from time import time

import logging
logger = logging.getLogger(__name__)

class NoiseSimplex(Shader):
    def use(self, speed: float, blend: float) -> None:
        if not self.allocated or not self.shader_program:
            logger.warning("NoiseSimplex shader not allocated or shader program missing.")
            return

        t: float = (time() * speed % (3600.0))

        # Activate shader program
        glUseProgram(self.shader_program)

        # Configure shader uniforms (using cached locations)
        glUniform1f(self.get_uniform_loc("time"), t)
        glUniform1f(self.get_uniform_loc("blend"), blend)

        # Render
        draw_quad()

