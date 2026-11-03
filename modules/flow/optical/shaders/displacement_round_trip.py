"""Displacement round trip shader: the forward displacement where the backward one brings it back."""

from OpenGL.GL import GL_TEXTURE0, GL_TEXTURE1, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1f, glUniform1i, glUseProgram
from modules.gl import Shader, draw_quad, Texture


class DisplacementRoundTrip(Shader):
    """Writes the forward displacement (RG, uv per frame), 0 where the round trip misses by more than `largest_miss`."""

    def use(self, forward: Texture, backward: Texture, largest_miss: float) -> None:
        """Take the round trip.

        Args:
            forward: uv per frame, the previous frame onto the current
            backward: uv per frame, the current frame onto the previous
            largest_miss: The most the round trip may miss by (px)
        """
        if not self.allocated or not self.shader_program or not forward.allocated or not backward.allocated:
            return

        glUseProgram(self.shader_program)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, forward.tex_id)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, backward.tex_id)

        glUniform1i(self.get_uniform_loc("forward"), 0)
        glUniform1i(self.get_uniform_loc("backward"), 1)
        glUniform1f(self.get_uniform_loc("largest_miss"), largest_miss)

        draw_quad()
