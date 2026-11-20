"""RGB to YUV shader: one plane of a screen frame's YUV (BT.709 video range), at 4:2:0 or 4:4:4, in any sample
size."""

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


def bt709_video_range(sample_max: int = 255) -> tuple[np.ndarray, np.ndarray]:
    """The matrix (rows Y, Cb, Cr on RGB 0..1) and offset that give BT.709 video-range codes as normalized samples
    of a plane whose samples go to `sample_max`: the 8-bit codes (Y 16 to 235, Cb and Cr 16 to 240 around 128) times
    (sample_max + 1) / 256, over sample_max. 255: 8 bits; 1023: 10 bits; 65535: 10 bits in the high bits of 16."""
    luma = np.array([KR, KG, KB])
    blue_difference = (np.array([0.0, 0.0, 1.0]) - luma) / (2.0 * (1.0 - KB))
    red_difference = (np.array([1.0, 0.0, 0.0]) - luma) / (2.0 * (1.0 - KR))
    scale = (sample_max + 1) / 256.0 / sample_max
    matrix = np.stack([luma * 219.0, blue_difference * 224.0, red_difference * 224.0]) * scale
    offset = np.array([16.0, 128.0, 128.0]) * scale
    return matrix.astype(np.float32), offset.astype(np.float32)


BT709: dict[int, tuple[np.ndarray, np.ndarray]] = {m: bt709_video_range(m) for m in (255, 1023, 65535)}


class YuvPlane(IntEnum):
    Y =    0    # at the frame's size, one channel
    CBCR = 1    # 4:2:0: Cb Cr interleaved at half width and half height, two channels
    CB =   2    # 4:4:4: at the frame's size, one channel
    CR =   3    # 4:4:4: at the frame's size, one channel
    CBYCR = 4   # 4:4:4 at 10 bits: Cb, Y, Cr in one RGB10_A2 texture at the frame's size, read back as xv30le


class RgbToYuv(Shader):
    """Draws one YUV plane of `src` into the bound target, sized as the plane is (`YuvPlane`); the target's first row
    is the picture's top row."""

    def use(self, src: Texture, plane: YuvPlane, sample_max: int = 255) -> None:
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
        matrix, offset = BT709[sample_max]
        glUniformMatrix3fv(self.get_uniform_loc("matrix"), 1, GL_FALSE, matrix)
        glUniform3fv(self.get_uniform_loc("offset"), 1, offset)
        draw_quad()
