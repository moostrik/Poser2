"""Velocity Event shader.

Peak-holds one event of the motion, a start or a stop, from the fast and slow trails.
"""

from enum import IntEnum

from OpenGL.GL import GL_TEXTURE0, GL_TEXTURE1, GL_TEXTURE2, GL_TEXTURE3, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1f, glUniform1i, glUseProgram
from modules.gl import Shader, draw_quad, Texture


class Event(IntEnum):
    START = 1       # fast - slow rises along the motion
    STOP = -1       # it falls


class VelocityEvent(Shader):
    """Writes max(event * gate, previous * decay) to the red channel."""

    def use(self, fast: Texture, slow: Texture, previous: Texture, event: Event, decay: float, patches: Texture, strength: float, lod: float) -> None:
        """Hold one event.

        Args:
            fast: RGB fast trail
            slow: RGB slow trail
            previous: Previous held event
            event: START or STOP
            decay: What remains of the held event this update
            patches: Fast speed in red and slow speed in green, with mipmaps
            strength: Gate strength, 0 = every event counts
            lod: Mip level of a patch
        """
        if not self.allocated or not self.shader_program:
            return
        if not fast.allocated or not slow.allocated or not previous.allocated:
            return

        glUseProgram(self.shader_program)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, fast.tex_id)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, slow.tex_id)
        glActiveTexture(GL_TEXTURE2)
        glBindTexture(GL_TEXTURE_2D, previous.tex_id)
        glActiveTexture(GL_TEXTURE3)
        glBindTexture(GL_TEXTURE_2D, patches.tex_id)

        glUniform1i(self.get_uniform_loc("tex0"), 0)
        glUniform1i(self.get_uniform_loc("tex1"), 1)
        glUniform1i(self.get_uniform_loc("tex2"), 2)
        glUniform1i(self.get_uniform_loc("tex3"), 3)
        glUniform1f(self.get_uniform_loc("sign"), float(event))
        glUniform1f(self.get_uniform_loc("decay"), decay)
        glUniform1f(self.get_uniform_loc("strength"), strength)
        glUniform1f(self.get_uniform_loc("lod"), lod)

        draw_quad()
