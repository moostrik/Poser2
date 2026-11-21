"""AdaptiveThreshold: a threshold set per pixel by its neighborhood's mean and variance."""

from OpenGL.GL import glUniform1f, glUniform1i, glUseProgram
from ... import Shader, draw_quad, Texture

import logging
logger = logging.getLogger(__name__)


class AdaptiveThreshold(Shader):
    """`src`'s luminance thresholded against its 3 x 3 neighborhood `radius` pixels apart; `detail_boost` keeps detail
    where the neighborhood varies; `invert` swaps black and white."""

    def use(self, src: Texture, threshold: float = 0.5, detail_boost: float = 1.0,
            radius: float = 5.0, invert: bool = False) -> None:
        if not self.allocated or not self.shader_program:
            logger.warning("AdaptiveThreshold shader not allocated or shader program missing.")
            return
        if not src.allocated:
            logger.warning("AdaptiveThreshold shader: input texture not allocated.")
            return

        glUseProgram(self.shader_program)
        src.bind(0)
        glUniform1f(self.get_uniform_loc("threshold"), threshold)
        glUniform1f(self.get_uniform_loc("detailBoost"), detail_boost)
        glUniform1f(self.get_uniform_loc("radius"), radius)
        glUniform1i(self.get_uniform_loc("invert"), 1 if invert else 0)
        draw_quad()
