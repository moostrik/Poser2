"""NV12 to RGB shader: an NV12 frame's Y and Cb Cr planes, BT.709 video range, back to RGB."""

import logging

import numpy as np
from OpenGL.GL import (GL_FALSE, GL_TEXTURE0, GL_TEXTURE1, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1i,
                       glUniform3fv, glUniformMatrix3fv, glUseProgram)

from modules.gl import Shader, draw_quad, Texture
from .nv12 import MATRIX, OFFSET

logger = logging.getLogger(__name__)

INVERSE: np.ndarray = np.linalg.inv(MATRIX.astype(np.float64)).astype(np.float32)     # rows R, G, B on Y Cb Cr


class Nv12ToRgb(Shader):
    """Draws the RGB picture of an NV12 frame into the bound target, upright: `luma` (R8) and `chroma` (RG8 at half
    size) hold its planes as uploaded, top row first."""

    def use(self, luma: Texture, chroma: Texture) -> None:
        if not self.allocated or not self.shader_program:
            logger.warning("Nv12ToRgb shader not allocated or shader program missing.")
            return
        if not luma.allocated or not chroma.allocated:
            logger.warning("Nv12ToRgb shader: input textures not allocated.")
            return

        glUseProgram(self.shader_program)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, luma.tex_id)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, chroma.tex_id)
        glActiveTexture(GL_TEXTURE0)
        glUniform1i(self.get_uniform_loc("luma"), 0)
        glUniform1i(self.get_uniform_loc("chroma"), 1)
        # Row-major rows read as GLSL columns, so `matrix[i]` in the shader is row i (R, G, B)
        glUniformMatrix3fv(self.get_uniform_loc("matrix"), 1, GL_FALSE, INVERSE)
        glUniform3fv(self.get_uniform_loc("offset"), 1, OFFSET)
        draw_quad()
