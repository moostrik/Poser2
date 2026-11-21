"""Blur: one pass of a separable Gaussian blur, across or down."""

from OpenGL.GL import glUniform1f, glUniform2f, glUseProgram
from ... import Shader, draw_quad, Texture

import logging
logger = logging.getLogger(__name__)

KERNEL_SIGMA: float = 9.0457    # the kernel's standard deviation in texels with its taps as written (blur.frag)


class Blur(Shader):
    """`src` blurred across (`horizontal`) or down, by a Gaussian of `sigma` pixels; the two passes in turn make the
    2D blur."""

    def use(self, src: Texture, horizontal: bool, sigma: float) -> None:
        if not self.allocated or not self.shader_program:
            logger.warning("Blur shader not allocated or shader program missing.")
            return
        if not src.allocated:
            logger.warning("Blur shader: input texture not allocated.")
            return

        glUseProgram(self.shader_program)
        src.bind(0)
        glUniform2f(self.get_uniform_loc("direction"), *((1.0, 0.0) if horizontal else (0.0, 1.0)))
        glUniform1f(self.get_uniform_loc("spacing"), sigma / KERNEL_SIGMA)
        draw_quad()
