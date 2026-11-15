"""SimplexNoise: fractal 3D simplex noise, animated in time, one octave set per channel."""

from OpenGL.GL import glUniform1f, glUseProgram
from ... import Shader, draw_quad

import logging
logger = logging.getLogger(__name__)


class SimplexNoise(Shader):
    """Animated fractal simplex noise at `frame_time` (the tick's time, s) times `speed`; `blend` zooms it."""

    def use(self, frame_time: float, speed: float, blend: float) -> None:
        if not self.allocated or not self.shader_program:
            logger.warning("SimplexNoise shader not allocated or shader program missing.")
            return

        t: float = frame_time * speed % 3600.0
        glUseProgram(self.shader_program)
        glUniform1f(self.get_uniform_loc("time"), t)
        glUniform1f(self.get_uniform_loc("blend"), blend)
        draw_quad()
