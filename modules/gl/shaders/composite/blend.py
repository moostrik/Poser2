"""Blend: a layer over a base in a blend mode GL's own blending cannot do, as `blend_modes.glsl` writes them."""

from enum import IntEnum

from OpenGL.GL import glUniform1f, glUniform1i, glUseProgram
from ... import Shader, draw_quad, Texture

import logging
logger = logging.getLogger(__name__)


class BlendShaderMode(IntEnum):
    """The blend modes of `blend_modes.glsl`, by its numbers. Normal, add, multiply and screen are GL's own as well
    (`BlendState.BlendMode`), which blends in the draw itself without a pass of its own."""
    NORMAL       = 0
    ADD          = 1
    MULTIPLY     = 2
    SCREEN       = 3
    DARKEN       = 10
    COLOR_BURN   = 11
    LINEAR_BURN  = 12
    LIGHTEN      = 13
    COLOR_DODGE  = 14
    OVERLAY      = 15
    SOFT_LIGHT   = 16
    HARD_LIGHT   = 17
    VIVID_LIGHT  = 18
    LINEAR_LIGHT = 19
    PIN_LIGHT    = 20
    HARD_MIX     = 21
    DIFFERENCE   = 22
    EXCLUSION    = 23
    SUBTRACT     = 24
    DIVIDE       = 25
    HUE          = 26
    SATURATION   = 27
    COLOR        = 28
    LUMINOSITY   = 29
    REFLECT      = 30
    GLOW         = 31


class Blend(Shader):
    """`src`, the layer, over `dst`, the base, in `mode`, at `amount` times the layer's alpha; the result's alpha is
    the layer's over the base's. A pass of its own, into a target that is neither input."""

    def use(self, src: Texture, dst: Texture, mode: BlendShaderMode, amount: float = 1.0) -> None:
        if not self.allocated or not self.shader_program:
            logger.warning("Blend shader not allocated or shader program missing.")
            return
        if not src.allocated or not dst.allocated:
            return
        glUseProgram(self.shader_program)
        src.bind(0)
        dst.bind(1)
        glUniform1i(self.get_uniform_loc("mode"), mode)
        glUniform1f(self.get_uniform_loc("amount"), amount)
        draw_quad()
