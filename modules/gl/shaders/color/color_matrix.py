"""ColorMatrix: every pixel's RGB through a 3 x 3 matrix and an offset; `saturation_matrix` keeps luminance."""

import numpy as np
from OpenGL.GL import GL_TEXTURE0, GL_TEXTURE_2D, GL_TRUE, glActiveTexture, glBindTexture, glUniform3f, glUniformMatrix3fv, glUseProgram
from ... import Shader, draw_quad, Texture

import logging
logger = logging.getLogger(__name__)

REC709: np.ndarray = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def saturation_matrix(saturation: float) -> np.ndarray:
    """The matrix that scales saturation by `saturation` and keeps Rec. 709 luminance: grey at 0, as is at 1."""
    grey: np.ndarray = np.tile(REC709, (3, 1))
    return (grey + saturation * (np.eye(3, dtype=np.float32) - grey)).astype(np.float32)


class ColorMatrix(Shader):
    """`src`'s RGB as `matrix @ rgb + offset` (rows give the output channels), alpha kept."""

    def use(self, src: Texture, matrix: np.ndarray, offset: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> None:
        if not self.allocated or not self.shader_program:
            logger.warning("ColorMatrix shader not allocated or shader program missing.")
            return
        if not src.allocated:
            return
        glUseProgram(self.shader_program)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, src.tex_id)
        glUniformMatrix3fv(self.get_uniform_loc("matrix"), 1, GL_TRUE, np.ascontiguousarray(matrix, dtype=np.float32))
        glUniform3f(self.get_uniform_loc("offset"), *offset)
        draw_quad()
