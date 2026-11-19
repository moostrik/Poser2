"""Inject Velocity 2D shader: the body's drag on the 2D velocity, or its body force added."""

from enum import IntEnum

from OpenGL.GL import GL_TEXTURE0, GL_TEXTURE1, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1f, glUniform1i, glUseProgram
from modules.gl import Shader, Texture, draw_quad


class VelocityInjection(IntEnum):
    """How the body goes into the velocity."""
    DRAG = 0    # the boundary: dragged toward
    ADD = 1     # the body force: added


class InjectVelocity2D(Shader):
    """Writes the velocity with the body's drag or its body force; draw into the velocity's other buffer."""

    def drag(self, velocity: Texture, source: Texture, gain: float, share: float, weight: float, floor: float) -> None:
        """The water pulled toward `source` × `gain` by `share` × its presence × `weight`, never past it."""
        self._use(velocity, source, VelocityInjection.DRAG, gain=gain, share=share, weight=weight, floor=floor)

    def add(self, velocity: Texture, source: Texture, strength: float) -> None:
        """`source` (an acceleration) × `strength` (weight × gain × dt) added."""
        self._use(velocity, source, VelocityInjection.ADD, strength=strength)

    def _use(self, velocity: Texture, source: Texture, mode: VelocityInjection, gain: float = 1.0, share: float = 0.0,
             weight: float = 1.0, floor: float = 1.0, strength: float = 0.0) -> None:
        if not self.allocated or not self.shader_program or not velocity.allocated or not source.allocated:
            return
        glUseProgram(self.shader_program)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, velocity.tex_id)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, source.tex_id)
        glUniform1i(self.get_uniform_loc("uVelocity"), 0)
        glUniform1i(self.get_uniform_loc("uSource"), 1)
        glUniform1i(self.get_uniform_loc("uMode"), int(mode))
        glUniform1f(self.get_uniform_loc("uGain"), gain)
        glUniform1f(self.get_uniform_loc("uShare"), share)
        glUniform1f(self.get_uniform_loc("uWeight"), weight)
        glUniform1f(self.get_uniform_loc("uFloor"), floor)
        glUniform1f(self.get_uniform_loc("uStrength"), strength)
        draw_quad()
