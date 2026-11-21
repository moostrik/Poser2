"""Z Position shader: where the body is now in the player's depth range."""

from OpenGL.GL import glUniform1f, glUniform1i, glUseProgram
from modules.gl import Shader, draw_quad, Texture


class ZPosition(Shader):
    """Writes the z position from the depth track times its weight to R, the weight to G."""

    def use(self, track: Texture, near: float, far: float, mask: Texture | None = None) -> None:
        """The z position.

        Args:
            track: The depth track (RGBA32F), its disparity (1/m) in green
            near: The range's front, m
            far: The range's back, m
            mask: A weight in R (0 to 1), None for 1
        """
        if not self.allocated or not self.shader_program or not track.allocated:
            return

        glUseProgram(self.shader_program)
        track.bind(0)
        if mask is not None:
            mask.bind(1)
        glUniform1i(self.get_uniform_loc("has_mask"), 1 if mask is not None else 0)
        glUniform1f(self.get_uniform_loc("near"), near)
        glUniform1f(self.get_uniform_loc("far"), far)
        draw_quad()
