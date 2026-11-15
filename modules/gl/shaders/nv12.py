"""NV12 shader: one plane of NV12 (Y, or interleaved Cb Cr at half size) from an RGB texture, BT.709 video range."""

import logging
from enum import IntEnum

import numpy as np
from OpenGL.GL import (GL_FALSE, GL_TEXTURE0, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1i,
                       glUniform3fv, glUniformMatrix3fv, glUseProgram)

from .. import Shader, draw_quad, Texture

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


class Nv12Plane(IntEnum):
    Y =  0
    UV = 1


class Nv12(Shader):
    """Draws one NV12 plane of `src` into the bound target: Y at the texture's size, UV at half its width and
    height; the target's first row is the picture's top row."""

    def use(self, src: Texture, plane: Nv12Plane) -> None:
        if not self.allocated or not self.shader_program:
            logger.warning("Nv12 shader not allocated or shader program missing.")
            return
        if not src.allocated:
            logger.warning("Nv12 shader: input texture not allocated.")
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
