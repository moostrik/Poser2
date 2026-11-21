"""Feather: a mask's edge softened inward, from the mask blurred."""

import logging

from OpenGL.GL import glUseProgram

from ... import Shader, draw_quad, Texture

logger = logging.getLogger(__name__)


class Feather(Shader):
    """`blurred`, a mask blurred by a Gaussian, made a feathered mask: 0 at the mask's edge and outside it, rising to
    1 inside over about two of the blur's sigmas, written to RGB."""

    def use(self, blurred: Texture) -> None:
        if not self.allocated or not self.shader_program:
            logger.warning("Feather shader not allocated or shader program missing.")
            return
        if not blurred.allocated:
            logger.warning("Feather shader: input texture not allocated.")
            return
        glUseProgram(self.shader_program)
        blurred.bind(0)
        draw_quad()
