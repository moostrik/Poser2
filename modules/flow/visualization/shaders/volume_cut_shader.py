"""Volume Cut shader.

A volume drawn flat: a slice at one layer, or its projection along one axis, the max or the mean.
"""

from OpenGL.GL import GL_TEXTURE0, GL_TEXTURE_3D, glActiveTexture, glBindTexture, glUniform1f, glUniform1i, glUseProgram
from modules.gl import Shader, Texture3D, draw_quad


class VolumeCutShader(Shader):
    """Draws a volume's slice or projection into the target bound, its channels as they are."""

    def use(self, volume: Texture3D, mode: int, position: float, use_max: bool, velocity: bool) -> None:
        """Draw the cut.

        Args:
            volume: The volume cut
            mode: 0 slice, 1 front, 2 top, 3 side
            position: The slice's place, 0 (front) to 1 (back)
            use_max: The projection's max, else its mean
            velocity: The volume is a velocity: its in-plane components into x and y
        """
        if not self.allocated or not self.shader_program:
            return
        if not volume.allocated:
            return

        glUseProgram(self.shader_program)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_3D, volume.tex_id)

        glUniform1i(self.get_uniform_loc("tex0"), 0)
        glUniform1i(self.get_uniform_loc("mode"), mode)
        glUniform1f(self.get_uniform_loc("position"), position)
        glUniform1i(self.get_uniform_loc("use_max"), 1 if use_max else 0)
        glUniform1i(self.get_uniform_loc("velocity"), 1 if velocity else 0)

        draw_quad()
        glBindTexture(GL_TEXTURE_3D, 0)
