"""Compose Velocity shader: a flow's x/y and z as one 3D velocity."""

from OpenGL.GL import glUniform1i, glUniform3f, glUseProgram
from modules.gl import Shader, draw_quad, Texture
from ...finite_guard import GuardSlot


class ComposeVelocity(Shader):
    """Writes (x, y, z) times strength to RGB, times the mask when given; 0 where it is not finite, counted by the
    guard."""

    def use(self, xy: Texture, z: Texture | None, strength: tuple[float, float, float], mask: Texture | None = None,
            guard: GuardSlot | None = None) -> None:
        """Compose the velocity.

        Args:
            xy: x and y in RG
            z: z in R, None without depth (z is 0)
            strength: Multiplier on x, y and z
            mask: Multiplier in R (0 to 1) on x, y and z, None for no mask
            guard: The velocity's slot in the motion's FiniteGuard; None no guard
        """
        if not self.allocated or not self.shader_program or not xy.allocated:
            return

        glUseProgram(self.shader_program)

        xy.bind(0)
        if z is not None:
            z.bind(1)
        if mask is not None:
            mask.bind(2)

        glUniform1i(self.get_uniform_loc("has_depth"), 1 if z is not None else 0)
        glUniform1i(self.get_uniform_loc("has_mask"), 1 if mask is not None else 0)
        glUniform3f(self.get_uniform_loc("strength"), *strength)
        if guard is not None:
            guard.use(self.get_uniform_loc("uGuardSlot"))
        else:
            glUniform1i(self.get_uniform_loc("uGuardSlot"), -1)

        draw_quad()
