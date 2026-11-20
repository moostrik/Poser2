"""YUV to RGB shader: a screen frame's Y and Cb Cr planes, BT.709 video range, at 4:2:0 or 4:4:4, back to RGB."""

import logging

import numpy as np
from OpenGL.GL import (GL_FALSE, GL_TEXTURE0, GL_TEXTURE1, GL_TEXTURE2, GL_TEXTURE_2D, glActiveTexture, glBindTexture,
                       glUniform1i, glUniform3fv, glUniformMatrix3fv, glUseProgram)

from modules.gl import Shader, draw_quad, Texture

from .rgb_to_yuv import MATRIX, OFFSET

logger = logging.getLogger(__name__)

INVERSE: np.ndarray = np.linalg.inv(MATRIX.astype(np.float64)).astype(np.float32)     # rows R, G, B on Y Cb Cr


class YuvToRgb(Shader):
    """Draws the RGB picture of a screen frame into the bound target, upright: `luma` (R8) and `chroma` hold its
    planes as uploaded, top row first; `chroma` is Cb Cr (RG8 at half size) for 4:2:0, or Cb (R8) with `cr` (R8)
    for 4:4:4."""

    def use(self, luma: Texture, chroma: Texture, cr: Texture | None = None) -> None:
        if not self.allocated or not self.shader_program:
            logger.warning("YuvToRgb shader not allocated or shader program missing.")
            return
        if not luma.allocated or not chroma.allocated or (cr is not None and not cr.allocated):
            logger.warning("YuvToRgb shader: input textures not allocated.")
            return

        glUseProgram(self.shader_program)
        for unit, texture in ((GL_TEXTURE0, luma), (GL_TEXTURE1, chroma), (GL_TEXTURE2, cr or chroma)):
            glActiveTexture(unit)
            glBindTexture(GL_TEXTURE_2D, texture.tex_id)
        glActiveTexture(GL_TEXTURE0)
        glUniform1i(self.get_uniform_loc("luma"), 0)
        glUniform1i(self.get_uniform_loc("chroma"), 1)
        glUniform1i(self.get_uniform_loc("cr"), 2)
        glUniform1i(self.get_uniform_loc("full_chroma"), int(cr is not None))
        # Row-major rows read as GLSL columns, so `matrix[i]` in the shader is row i (R, G, B)
        glUniformMatrix3fv(self.get_uniform_loc("matrix"), 1, GL_FALSE, INVERSE)
        glUniform3fv(self.get_uniform_loc("offset"), 1, OFFSET)
        draw_quad()
