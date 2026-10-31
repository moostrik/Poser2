"""Kinetic Reservoir shader.

Ink stored per pixel: filled by motion, leaking, drained by what a release emits.
"""

from OpenGL.GL import GL_TEXTURE0, GL_TEXTURE1, GL_TEXTURE2, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1f, glUniform1i, glUseProgram
from modules.gl import Shader, draw_quad, Texture


class KineticReservoir(Shader):
    """Writes the next reservoir to the red channel."""

    def use(self, previous: Texture, fast: Texture, held_release: Texture, leak: float, dt: float, gain: float) -> None:
        """Advance the reservoir.

        Args:
            previous: Previous reservoir
            fast: RG fast trail
            held_release: Held release in the red channel
            leak: What remains after the leak this update
            dt: Seconds this update
            gain: Emission gain
        """
        if not self.allocated or not self.shader_program:
            return
        if not previous.allocated or not fast.allocated or not held_release.allocated:
            return

        glUseProgram(self.shader_program)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, previous.tex_id)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, fast.tex_id)
        glActiveTexture(GL_TEXTURE2)
        glBindTexture(GL_TEXTURE_2D, held_release.tex_id)

        glUniform1i(self.get_uniform_loc("tex0"), 0)
        glUniform1i(self.get_uniform_loc("tex1"), 1)
        glUniform1i(self.get_uniform_loc("tex2"), 2)
        glUniform1f(self.get_uniform_loc("leak"), leak)
        glUniform1f(self.get_uniform_loc("dt"), dt)
        glUniform1f(self.get_uniform_loc("gain"), gain)

        draw_quad()
