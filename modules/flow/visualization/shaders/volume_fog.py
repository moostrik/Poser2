"""Volume Fog shader.

A volume ray-marched through its box from the orbit camera, each sample colored as the channel view
colors it and its amount its opacity.
"""

import numpy as np
from OpenGL.GL import (GL_TEXTURE0, GL_TEXTURE_3D, GL_TRUE, glActiveTexture, glBindTexture, glUniform1f, glUniform1i,
                       glUniform3f, glUniformMatrix4fv, glUseProgram)
from modules.gl import Shader, Texture3D, draw_quad


class VolumeFogShader(Shader):
    """Draws the fog over the target bound, premultiplied."""

    def use(self, volume: Texture3D, inverse_view_projection: np.ndarray, box: tuple[float, float, float],
            scale: float, channel: int, colormap: bool, steps: int, density: float) -> None:
        """Draw the fog.

        Args:
            volume: The volume
            inverse_view_projection: Clip space back to the box's space (4 × 4, row-major)
            box: The box's size, its width 1
            scale: Multiplier on the values
            channel: 0 to 3 for x, y, z, w; 4 for all
            colormap: Centered on grey, instead of as stored
            steps: Samples along a ray through the whole box
            density: Opacity per box width at an amount of 1
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
        glUniform1i(self.get_uniform_loc("colormap"), 1 if colormap else 0)
        glUniform1i(self.get_uniform_loc("steps"), steps)
        glUniform1f(self.get_uniform_loc("density"), density)
        draw_quad()
        glBindTexture(GL_TEXTURE_3D, 0)
