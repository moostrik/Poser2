"""RGB to YUV shader: one plane of a screen frame's YUV (BT.709 video range), at 4:2:0 or 4:4:4."""

import logging
from enum import IntEnum

import numpy as np
from OpenGL.GL import (GL_FALSE, GL_TEXTURE0, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1i,
                       glUniform3fv, glUniformMatrix3fv, glUseProgram)

from modules.gl import Shader, draw_quad, Texture

logger = logging.getLogger(__name__)

# BT.709 luma weights
KR: float = 0.2126
KB: float = 0.0722
KG: float = 1.0 - KR - KB


def bt709_video_range() -> tuple[np.ndarray, np.ndarray]:
    """The matrix (rows Y, Cb, Cr on RGB 0..1) and offset that give BT.709 video-range codes over 255:
    Y 16 to 235, Cb and Cr 16 to 240 around 128."""
    luma = np.array([KR, KG, KB])
    blue_difference = (np.array([0.0, 0.0, 1.0]) - luma) / (2.0 * (1.0 - KB))
    red_difference = (np.array([1.0, 0.0, 0.0]) - luma) / (2.0 * (1.0 - KR))
    matrix = np.stack([luma * 219.0, blue_difference * 224.0, red_difference * 224.0]) / 255.0
    offset = np.array([16.0, 128.0, 128.0]) / 255.0
    return matrix.astype(np.float32), offset.astype(np.float32)


MATRIX, OFFSET = bt709_video_range()


class YuvPlane(IntEnum):
    Y =    0    # at the frame's size, one channel
    CBCR = 1    # 4:2:0: Cb Cr interleaved at half width and half height, two channels
    CB =   2    # 4:4:4: at the frame's size, one channel
    CR =   3    # 4:4:4: at the frame's size, one channel


class RgbToYuv(Shader):
    """Draws one YUV plane of `src` into the bound target, sized as the plane is (`YuvPlane`); the target's first row
    is the picture's top row."""

    def use(self, src: Texture, plane: YuvPlane) -> None:
        if not self.allocated or not self.shader_program:
            logger.warning("RgbToYuv shader not allocated or shader program missing.")
            return
        if not src.allocated:
            logger.warning("RgbToYuv shader: input texture not allocated.")
            return

        glUseProgram(self.shader_program)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, src.tex_id)
        glUniform1i(self.get_uniform_loc("src"), 0)
        glUniform1i(self.get_uniform_loc("plane"), int(plane))
        # Row-major rows read as GLSL columns, so `matrix[i]` in the shader is row i (Y, Cb, Cr)
        glUniformMatrix3fv(self.get_uniform_loc("matrix"), 1, GL_FALSE, MATRIX)
        glUniform3fv(self.get_uniform_loc("offset"), 1, OFFSET)
        draw_quad()
