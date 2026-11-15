from OpenGL.GL import glUniform1f, glUseProgram
from ... import Shader, draw_quad

import logging
logger = logging.getLogger(__name__)

class NoiseSimplex(Shader):
    def use(self, frame_time: float, speed: float, blend: float) -> None:
        """Animated simplex noise at `frame_time` (the tick's time, s) times `speed`."""
        if not self.allocated or not self.shader_program:
            logger.warning("NoiseSimplex shader not allocated or shader program missing.")
            return

        t: float = (frame_time * speed % (3600.0))

        # Activate shader program
        glUseProgram(self.shader_program)

        # Configure shader uniforms (using cached locations)
        glUniform1f(self.get_uniform_loc("time"), t)
        glUniform1f(self.get_uniform_loc("blend"), blend)

        # Render
        draw_quad()
