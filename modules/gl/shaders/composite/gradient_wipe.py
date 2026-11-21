"""GradientWipe: a wipe from one image to another in the order a gradient image gives."""

from OpenGL.GL import glUniform1f, glUseProgram
from ... import Shader, draw_quad, Texture

import logging
logger = logging.getLogger(__name__)


class GradientWipe(Shader):
    """`src` turns into `dst` where `gradient`'s red is passed by `progress` (0..1): dark first, light last;
    `softness` (0..1) widens the edge, 0 a hard one."""

    def use(self, src: Texture, dst: Texture, gradient: Texture, progress: float, softness: float = 0.0) -> None:
        if not self.allocated or not self.shader_program:
            logger.warning("GradientWipe shader not allocated or shader program missing.")
            return
        if not src.allocated or not dst.allocated or not gradient.allocated:
            logger.warning("GradientWipe shader: input textures not allocated.")
            return

        glUseProgram(self.shader_program)
        for unit, texture in enumerate((src, dst, gradient)):
            texture.bind(unit)
        glUniform1f(self.get_uniform_loc("progress"), progress)
        glUniform1f(self.get_uniform_loc("softness"), softness)
        draw_quad()
