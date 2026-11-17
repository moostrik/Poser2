"""LucasKanadeTile - one mip level of the Lucas–Kanade solve, a tile of texels a workgroup with the current frame shared."""
from __future__ import annotations

import math

from OpenGL.GL import GL_RG32F, GL_TEXTURE_2D, GL_WRITE_ONLY, glActiveTexture, glBindTexture, glDispatchCompute, \
    glUniform1f, glUniform1i, glUseProgram, GL_TEXTURE0

from modules.gl import ComputeShader, Texture


class LucasKanadeTile(ComputeShader):
    """The Lucas–Kanade solve at one mip level, its window's reads of the current frame from shared memory."""

    TILE: int = 16

    def use(self, target: Texture, previous: Texture, current: Texture, estimate: Texture | None, level: int,
            window: int, damping: float) -> None:
        """Refine `estimate` at mip `level` into `target` (RG32F at the level's size).

        Args:
            target: The level's solve, written
            previous: Previous luminance frame, mipmapped
            current: Current luminance frame, mipmapped
            estimate: The displacement from the coarser levels, uv per frame; None at the coarsest, which starts from no motion
            level: The mip level to solve at
            window: Window radius in texels of that level
            damping: Added to the window's gradient sums, so a flat window solves to no motion
        """
        if not self.allocated or not self.shader_program:
            return
        if not previous.allocated or not current.allocated or (estimate is not None and not estimate.allocated):
            return
        glUseProgram(self.shader_program)
        for unit, texture in enumerate((previous, current, estimate)):
            glActiveTexture(int(GL_TEXTURE0) + unit)
            glBindTexture(GL_TEXTURE_2D, texture.tex_id if texture is not None else 0)
        glUniform1i(self.get_uniform_loc("has_estimate"), int(estimate is not None))
        glUniform1i(self.get_uniform_loc("level"), level)
        glUniform1i(self.get_uniform_loc("window"), window)
        glUniform1f(self.get_uniform_loc("damping"), damping)
        self.bind_image(0, target, GL_WRITE_ONLY, GL_RG32F)
        glDispatchCompute(math.ceil(target.width / self.TILE), math.ceil(target.height / self.TILE), 1)
