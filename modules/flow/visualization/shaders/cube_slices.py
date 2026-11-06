"""Cube Slices shader.

A volume's layers as stacked quads in its box, in one instanced draw, far to near.
"""

import numpy as np
from OpenGL.GL import (GL_TEXTURE0, GL_TEXTURE_3D, GL_TRIANGLE_FAN, GL_TRUE, glActiveTexture, glBindTexture,
                       glDrawArraysInstanced, glUniform1f, glUniform1i, glUniform3f, glUniformMatrix4fv, glUseProgram)
from modules.gl import Shader, Texture3D


class CubeSlices(Shader):
    """Draws every layer of a volume at its depth, the selected one pulled out of the stack."""

    def use(self, volume: Texture3D, view_projection: np.ndarray, box: tuple[float, float, float],
            back_first: bool, selected: int, gap: float, scale: float, channel: int, colormap: bool,
            opacity: float) -> None:
        """Draw the stack.

        Args:
            volume: The volume drawn
            view_projection: The box's space to clip space (4 × 4, row-major)
            box: The box's size, its width 1
            back_first: The camera is on the front's side: draw the back layers first
            selected: The layer pulled out and drawn opaque
            gap: The gap on each side of it, in layer spacings
            scale: Multiplier on the values
            channel: 0 to 3 for x, y, z, w; 4 for all
            colormap: Centered on mid grey, instead of as stored
            opacity: The other layers' opacity
        """
        if not self.allocated or not self.shader_program or not volume.allocated:
            return

        glUseProgram(self.shader_program)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_3D, volume.tex_id)

        glUniform1i(self.get_uniform_loc("tex0"), 0)
        glUniformMatrix4fv(self.get_uniform_loc("view_projection"), 1, GL_TRUE, view_projection.astype(np.float32))
        glUniform3f(self.get_uniform_loc("box"), *box)
        glUniform1i(self.get_uniform_loc("layers"), volume.depth)
        glUniform1i(self.get_uniform_loc("back_first"), 1 if back_first else 0)
        glUniform1i(self.get_uniform_loc("selected"), selected)
        glUniform1f(self.get_uniform_loc("gap"), gap)
        glUniform1f(self.get_uniform_loc("scale"), scale)
        glUniform1i(self.get_uniform_loc("channel"), channel)
        glUniform1i(self.get_uniform_loc("colormap"), 1 if colormap else 0)
        glUniform1f(self.get_uniform_loc("opacity"), opacity)

        glDrawArraysInstanced(GL_TRIANGLE_FAN, 0, 4, volume.depth)
        glBindTexture(GL_TEXTURE_3D, 0)
