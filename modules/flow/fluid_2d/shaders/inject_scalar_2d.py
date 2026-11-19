"""Inject Scalar 2D shader: a scalar input added to a 2D field, or the field filled toward it."""

from enum import IntEnum

from OpenGL.GL import GL_TEXTURE0, GL_TEXTURE1, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1f, glUniform1i, glUseProgram
from modules.gl import Shader, Texture, draw_quad


class ScalarInjection(IntEnum):
    """How a scalar goes into its field."""
    ADD = 0     # added: the expansion's
    FILL = 1    # dragged toward, never toward a fainter body: the temperature's


class InjectScalar2D(Shader):
    """Writes the field with the input added or filled toward; draw into the field's other buffer."""

    def use(self, field: Texture, input: Texture, mode: ScalarInjection, strength: float, gain: float = 1.0) -> None:
        """Inject the scalar.

        Args:
            field: The field before (red)
            input: The scalar (red, signed)
            mode: Add or fill
            strength: Add: the weight; fill: the drag's share × the weight
            gain: Fill: the input times this, at most ±1, is the target
        """
        if not self.allocated or not self.shader_program or not field.allocated or not input.allocated:
            return
        glUseProgram(self.shader_program)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, field.tex_id)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, input.tex_id)
        glUniform1i(self.get_uniform_loc("uField"), 0)
        glUniform1i(self.get_uniform_loc("uInput"), 1)
        glUniform1i(self.get_uniform_loc("uMode"), int(mode))
        glUniform1f(self.get_uniform_loc("uStrength"), strength)
        glUniform1f(self.get_uniform_loc("uGain"), gain)
        draw_quad()
