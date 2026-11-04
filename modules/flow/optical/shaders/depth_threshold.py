"""Depth Threshold shader.

Takes the noise floor off the motion toward the camera; with a displacement, only where the image moves.
"""

from OpenGL.GL import GL_TEXTURE0, GL_TEXTURE1, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1f, glUniform1i, glUseProgram
from modules.gl import Shader, draw_quad, Texture


class DepthThreshold(Shader):
    """Writes the field with `threshold` taken off its size to the red channel."""

    def use(self, field: Texture, threshold: float, displacement: Texture | None = None, motion_floor: float = 0.0) -> None:
        """Take the noise floor off.

        Args:
            field: Signed field in the red channel
            threshold: Size taken off, in the field's units
            displacement: uv per frame; when given, the field is 0 where the image does not move
            motion_floor: Displacement the image must move by, in its 3×3 (px per frame)
        """
        if not self.allocated or not self.shader_program:
            return
        if not field.allocated:
            return

        glUseProgram(self.shader_program)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, field.tex_id)
        gate: bool = displacement is not None and displacement.allocated
        if gate:
            assert displacement is not None
            glActiveTexture(GL_TEXTURE1)
            glBindTexture(GL_TEXTURE_2D, displacement.tex_id)

        glUniform1i(self.get_uniform_loc("tex0"), 0)
        glUniform1i(self.get_uniform_loc("displacement"), 1)
        glUniform1f(self.get_uniform_loc("threshold"), threshold)
        glUniform1i(self.get_uniform_loc("gate"), int(gate))
        glUniform1f(self.get_uniform_loc("motion_floor"), motion_floor)

        draw_quad()
