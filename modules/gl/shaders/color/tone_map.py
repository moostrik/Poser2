"""ToneMap: high-range color to the display's 0..1, by a textbook operator."""

from enum import IntEnum

from OpenGL.GL import glUniform1f, glUniform1i, glUseProgram
from ... import Shader, draw_quad, Texture

import logging
logger = logging.getLogger(__name__)


class ToneMapOperator(IntEnum):
    """How high values are brought into 0..1."""
    REINHARD = 0    # c / (1 + c), per channel
    ACES     = 1    # Narkowicz's fit of the ACES filmic curve


class ToneMap(Shader):
    """`src` times `exposure`, through `operator`, alpha kept."""

    def use(self, src: Texture, operator: ToneMapOperator = ToneMapOperator.ACES, exposure: float = 1.0) -> None:
        if not self.allocated or not self.shader_program:
            logger.warning("ToneMap shader not allocated or shader program missing.")
            return
        if not src.allocated:
            return
        glUseProgram(self.shader_program)
        src.bind(0)
        glUniform1i(self.get_uniform_loc("operator"), operator)
        glUniform1f(self.get_uniform_loc("exposure"), exposure)
        draw_quad()
