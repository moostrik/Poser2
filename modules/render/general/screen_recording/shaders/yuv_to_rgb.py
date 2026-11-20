"""YUV to RGB shader: a screen frame's YUV planes, BT.709 video range, at 4:2:0 or 4:4:4, in any sample size, back to
RGB."""

import logging

import numpy as np
from OpenGL.GL import (GL_FALSE, GL_TEXTURE0, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1i,
                       glUniform3fv, glUniformMatrix3fv, glUseProgram)

from modules.gl import Shader, draw_quad, Texture

from .rgb_to_yuv import BT709

logger = logging.getLogger(__name__)

# per sample maximum: rows R, G, B on Y Cb Cr, the inverse of `RgbToYuv`'s matrix
INVERSE: dict[int, np.ndarray] = {sample_max: np.linalg.inv(matrix.astype(np.float64)).astype(np.float32)
                                  for sample_max, (matrix, _) in BT709.items()}


class YuvToRgb(Shader):
    """Draws the RGB picture of a screen frame into the bound target, upright, from its planes as uploaded, top row
    first: Y and Cb Cr at half size (4:2:0); Y, Cb and Cr (4:4:4); or one plane of Cb, Y, Cr (4:4:4 packed). Their
    samples go to `sample_max`."""

    def use(self, planes: list[Texture], sample_max: int = 255) -> None:
        if not self.allocated or not self.shader_program:
            logger.warning("YuvToRgb shader not allocated or shader program missing.")
            return
        if not 1 <= len(planes) <= 3 or not all(plane.allocated for plane in planes):
            logger.warning("YuvToRgb shader: input textures not allocated.")
            return

        glUseProgram(self.shader_program)
        for unit in range(3):
            glActiveTexture(int(GL_TEXTURE0) + unit)
            glBindTexture(GL_TEXTURE_2D, planes[min(unit, len(planes) - 1)].tex_id)
            glUniform1i(self.get_uniform_loc(f"plane{unit}"), unit)
        glActiveTexture(GL_TEXTURE0)
        glUniform1i(self.get_uniform_loc("num_planes"), len(planes))
        # Row-major rows read as GLSL columns, so `matrix[i]` in the shader is row i (R, G, B)
        glUniformMatrix3fv(self.get_uniform_loc("matrix"), 1, GL_FALSE, INVERSE[sample_max])
        glUniform3fv(self.get_uniform_loc("offset"), 1, BT709[sample_max][1])
        draw_quad()
