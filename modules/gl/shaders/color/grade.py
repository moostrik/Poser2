"""Grade: an image's white balance, exposure and offset, gamma, contrast around mid grey and saturation, in that
order."""

import logging

import numpy as np
from OpenGL.GL import (GL_TEXTURE0, GL_TEXTURE_2D, GL_TRUE, glActiveTexture, glBindTexture, glUniform1f, glUniform1i,
                       glUniformMatrix3fv, glUseProgram)

from ... import Shader, draw_quad, Texture

logger = logging.getLogger(__name__)


class Grade(Shader):
    """`src` graded: its linear light times `white_balance` (3 × 3, row-major), times `exposure` plus `offset`, to the power 1 / `gamma`, its contrast scaled by `contrast`
    around 0.5 and clamped, its saturation (HSV) by `saturation`; alpha kept."""

    def use(self, src: Texture, white_balance: np.ndarray, exposure: float, offset: float, gamma: float, contrast: float, saturation: float) -> None:
        if not self.allocated or not self.shader_program:
            logger.warning("Grade shader not allocated or shader program missing.")
            return
        if not src.allocated:
            logger.warning("Grade shader: input texture not allocated.")
            return
        glUseProgram(self.shader_program)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, src.tex_id)
        glUniform1i(self.get_uniform_loc("src"), 0)
        glUniformMatrix3fv(self.get_uniform_loc("white_balance"), 1, GL_TRUE, white_balance)
        glUniform1f(self.get_uniform_loc("exposure"), exposure)
        glUniform1f(self.get_uniform_loc("offset"), offset)
        glUniform1f(self.get_uniform_loc("gamma"), gamma)
        glUniform1f(self.get_uniform_loc("contrast"), contrast)
        glUniform1f(self.get_uniform_loc("saturation"), saturation)
        draw_quad()
