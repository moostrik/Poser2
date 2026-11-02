"""Kinetic Advect shader.

Moves a field along the fast trail's velocity.
"""

from OpenGL.GL import GL_TEXTURE0, GL_TEXTURE1, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1i, glUniform2f, glUseProgram
from modules.gl import Shader, draw_quad, Texture


class KineticAdvect(Shader):
    """Writes the field taken from where the fast trail says each pixel came from."""

    def use(self, field: Texture, fast: Texture, carry: tuple[float, float]) -> None:
        """Move the field.

        Args:
            field: Field to move
            fast: Fast trail, whose xy carries it
            carry: uv per frame per unit of trail velocity, times the advect amount
        """
        if not self.allocated or not self.shader_program:
            return
        if not field.allocated or not fast.allocated:
            return

        glUseProgram(self.shader_program)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, field.tex_id)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, fast.tex_id)

        glUniform1i(self.get_uniform_loc("tex0"), 0)
        glUniform1i(self.get_uniform_loc("tex1"), 1)
        glUniform2f(self.get_uniform_loc("carry"), carry[0], carry[1])

        draw_quad()
