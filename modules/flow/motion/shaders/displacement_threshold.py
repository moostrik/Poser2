"""Displacement threshold shader: the noise floor taken off a displacement."""

from OpenGL.GL import glUniform1f, glUseProgram
from modules.gl import Shader, draw_quad, Texture


class DisplacementThreshold(Shader):
    """Writes the displacement (RG, uv per frame) with `threshold` pixels taken off its length."""

    def use(self, displacement: Texture, threshold: float) -> None:
        """Take the noise floor off.

        Args:
            displacement: uv per frame
            threshold: Taken off the length (px per frame)
        """
        if not self.allocated or not self.shader_program or not displacement.allocated:
            return

        glUseProgram(self.shader_program)

        displacement.bind(0)

        glUniform1f(self.get_uniform_loc("threshold"), threshold)

        draw_quad()
