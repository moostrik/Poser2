"""LucasKanadeTile - one mip level of the Lucas–Kanade solve, a tile of texels a workgroup with the current frame
and the depth shared."""
from __future__ import annotations

import math

from OpenGL.GL import GL_RG32F, GL_TEXTURE_2D, GL_WRITE_ONLY, glActiveTexture, glBindTexture, glDispatchCompute, \
    glUniform1f, glUniform1i, glUseProgram, GL_TEXTURE0

from modules.gl import ComputeShader, Texture


class LucasKanadeTile(ComputeShader):
    """The Lucas–Kanade solve at one mip level, its window's reads of the current frame and the depth from shared memory."""

    TILE: int = 16

    def use(self, target: Texture, previous: Texture, current: Texture, estimate: Texture | None, level: int,
            window: int, damping: float, depth: Texture | None = None, tolerance: float = 0.0) -> None:
        """Refine `estimate` at mip `level` into `target` (RG32F at the level's size).

        Args:
            target: The level's solve, written
            previous: Previous luminance frame, mipmapped
            current: Current luminance frame, mipmapped
            estimate: The displacement from the coarser levels, uv per frame; None at the coarsest, which starts from no motion
            level: The mip level to solve at
            window: Window radius in texels of that level
            damping: Added to the window's gradient sums, so a flat window solves to no motion
            depth: Metres (R); when given, only the samples within `tolerance` of the centre's depth count
            tolerance: Same-surface depth difference (m), with depth
        """
        if not self.allocated or not self.shader_program:
            return
        if not previous.allocated or not current.allocated or (estimate is not None and not estimate.allocated):
            return
        glUseProgram(self.shader_program)
        same_surface: bool = depth is not None and depth.allocated
        for unit, texture in enumerate((previous, current, estimate, depth if same_surface else None)):
            glActiveTexture(int(GL_TEXTURE0) + unit)
            glBindTexture(GL_TEXTURE_2D, texture.tex_id if texture is not None else 0)
        glUniform1i(self.get_uniform_loc("has_estimate"), int(estimate is not None))
        glUniform1i(self.get_uniform_loc("level"), level)
        glUniform1i(self.get_uniform_loc("window"), window)
        glUniform1f(self.get_uniform_loc("damping"), damping)
        glUniform1i(self.get_uniform_loc("same_surface"), int(same_surface))
        glUniform1f(self.get_uniform_loc("tolerance"), tolerance)
        self.bind_image(0, target, GL_WRITE_ONLY, GL_RG32F)
        glDispatchCompute(math.ceil(target.width / self.TILE), math.ceil(target.height / self.TILE), 1)
