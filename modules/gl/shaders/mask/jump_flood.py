"""JumpFlood: the jump flooding algorithm's passes (Rong and Tan 2006), a mask's nearest-pixel map in log2 passes."""

from OpenGL.GL import glUniform1f, glUniform1i, glUseProgram
from ... import Shader, draw_quad, Texture

import logging
logger = logging.getLogger(__name__)

SEED: int = 0
STEP: int = 1
DISTANCE: int = 2


class JumpFlood(Shader):
    """The passes `DistanceField` chains: `seed` marks the mask's pixels, each `step` looks `jump` pixels away for a
    nearer one, `distance` writes the distance to it."""

    def seed(self, mask: Texture, threshold: float = 0.5) -> None:
        """Every pixel whose red reaches `threshold`: its own coordinates; the rest: none (-1)."""
        self._use(SEED, mask, threshold=threshold)

    def step(self, nearest: Texture, jump: int) -> None:
        """Each pixel's nearest seed among its own and its eight neighbors' `jump` pixels away."""
        self._use(STEP, nearest, jump=jump)

    def distance(self, nearest: Texture) -> None:
        """The distance in pixels to the nearest seed, in red; -1 without a seed."""
        self._use(DISTANCE, nearest)

    def _use(self, stage: int, src: Texture, threshold: float = 0.5, jump: int = 1) -> None:
        if not self.allocated or not self.shader_program:
            logger.warning("JumpFlood shader not allocated or shader program missing.")
            return
        if not src.allocated:
            return
        glUseProgram(self.shader_program)
        src.bind(0)
        glUniform1i(self.get_uniform_loc("stage"), stage)
        glUniform1f(self.get_uniform_loc("threshold"), threshold)
        glUniform1i(self.get_uniform_loc("jump"), jump)
        draw_quad()
