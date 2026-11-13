"""Volume Cube shader.

A volume ray-marched through its box from the orbit camera, each sample colored by the mode and its
amount its opacity; with the slice on, the selected layer read solid and the rest dimmed.
"""

from enum import IntEnum

import numpy as np
from OpenGL.GL import (GL_TEXTURE0, GL_TEXTURE_3D, GL_TRUE, glActiveTexture, glBindTexture, glUniform1f, glUniform1i,
                       glUniform3f, glUniformMatrix4fv, glUseProgram)
from modules.gl import Shader, Texture3D, draw_quad


class CubeColor(IntEnum):
    """How the cube colors a sample."""
    AS_STORED = 0           # the values as they are: an image's color, or raw
    DIRECTION = 1           # a vector by its direction alone
    CENTERED = 2            # a scalar centered on grey, so the sign shows


class VolumeCubeShader(Shader):
    """Draws the cube over the target bound, premultiplied."""

    def use(self, volume: Texture3D, inverse_view_projection: np.ndarray, box: tuple[float, float, float],
            scale: float, channel: int, color: CubeColor, steps: int, opacity: float, depth_fade: float,
            slice_z: float | None, slice_focus: float = 0.5) -> None:
        """Draw the cube.

        Args:
            volume: The volume
            inverse_view_projection: Clip space back to the box's space (4 × 4, row-major)
            box: The box's size, its width 1
            scale: Multiplier on the values; at 1 after it, a sample is full
            channel: 0 to 3 for x, y, z, w; 4 for all
            color: How a sample is colored
            steps: Samples along a ray through the whole box
            opacity: Opacity per box width of a full sample
            depth_fade: Dimming per box width behind the box's near side
            slice_z: The selected layer's depth, 0 front to 1 back, in focus; None draws no slice
            slice_focus: 0 the slice as any sample, 1 the slice at its amount and the rest at none
        """
        if not self.allocated or not self.shader_program or not volume.allocated:
            return
        glUseProgram(self.shader_program)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_3D, volume.tex_id)
        glUniform1i(self.get_uniform_loc("tex0"), 0)
        glUniformMatrix4fv(self.get_uniform_loc("inverse_view_projection"), 1, GL_TRUE, inverse_view_projection.astype(np.float32))
        glUniform3f(self.get_uniform_loc("box"), *box)
        glUniform1f(self.get_uniform_loc("scale"), scale)
        glUniform1i(self.get_uniform_loc("channel"), channel)
        glUniform1i(self.get_uniform_loc("mode"), int(color))
        glUniform1i(self.get_uniform_loc("steps"), steps)
        glUniform1f(self.get_uniform_loc("opacity"), opacity)
        glUniform1f(self.get_uniform_loc("depth_fade"), depth_fade)
        glUniform1i(self.get_uniform_loc("slice_on"), 0 if slice_z is None else 1)
        glUniform1f(self.get_uniform_loc("slice_z"), 0.5 if slice_z is None else slice_z)
        glUniform1f(self.get_uniform_loc("slice_focus"), slice_focus)
        draw_quad()
        glBindTexture(GL_TEXTURE_3D, 0)
