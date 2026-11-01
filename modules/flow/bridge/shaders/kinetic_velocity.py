"""Kinetic Velocity shader.

Blends the velocity output from the fast trail toward the slow trail where the motion releases.
"""

from OpenGL.GL import GL_TEXTURE0, GL_TEXTURE1, GL_TEXTURE2, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1f, glUniform1i, glUseProgram
from modules.gl import Shader, draw_quad, Texture


class KineticVelocity(Shader):
    """Writes the xy of mix(fast, slow, drop * release weight) to RG, or its z to R."""

    def use(self, fast: Texture, slow: Texture, acceleration: Texture, drop: float, depth: bool = False) -> None:
        """Compose the velocity output.

        Args:
            fast: RGB fast trail
            slow: RGB slow trail
            acceleration: Signed acceleration in the red channel
            drop: How far a full release blends to the slow trail, 0..1
            depth: Write the z to red instead of the xy to RG
        """
        if not self.allocated or not self.shader_program:
            return
        if not fast.allocated or not slow.allocated or not acceleration.allocated:
            return

        glUseProgram(self.shader_program)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, fast.tex_id)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, slow.tex_id)
        glActiveTexture(GL_TEXTURE2)
        glBindTexture(GL_TEXTURE_2D, acceleration.tex_id)

        glUniform1i(self.get_uniform_loc("tex0"), 0)
        glUniform1i(self.get_uniform_loc("tex1"), 1)
        glUniform1i(self.get_uniform_loc("tex2"), 2)
        glUniform1f(self.get_uniform_loc("drop"), drop)
        glUniform1i(self.get_uniform_loc("depth"), 1 if depth else 0)

        draw_quad()
