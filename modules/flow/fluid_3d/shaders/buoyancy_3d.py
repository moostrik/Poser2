"""Buoyancy3D - the buoyancy force: the density's weight along gravity, the temperature's lift against it."""
from __future__ import annotations

from OpenGL.GL import GL_RGBA16F, glUniform1f, glUniform1i, glUniform3f, glUniform3i, glUseProgram

from modules.gl import Texture3D, ComputeShader


class Buoyancy3D(ComputeShader):
    """Compute the 3D buoyancy force: F = gravity * (κ·density − σ·T), T the warmth, 0 the room's."""

    WORKGROUP_SIZE_X = 8
    WORKGROUP_SIZE_Y = 8
    WORKGROUP_SIZE_Z = 4

    def __init__(self) -> None:
        super().__init__()

    def use(self, temperature: Texture3D, density: Texture3D,
            obstacle: Texture3D, force_out: Texture3D,
            sigma: float, kappa: float,
            gravity: tuple[float, float, float] = (0.0, -1.0, 0.0),
            has_obstacles: bool = True) -> None:
        """Compute the buoyancy force.

        Args:
            temperature: 3D warmth field (R16F), −1 to 1, 0 the room's
            density: 3D density field (RGBA16F; its weight is its amount, alpha)
            obstacle: 3D obstacle mask (R8)
            force_out: Output force volume (RGBA16F)
            sigma: Thermal buoyancy (widths/s² per unit of warmth)
            kappa: Density weight (widths/s² per unit of density)
            gravity: Gravity direction in the grid's frame, y up — (0,-1,0) down the screen, (0,0,1) away from viewer
            has_obstacles: Whether obstacle logic is active
        """
        if not self.allocated or not self.shader_program:
            return

        glUseProgram(self.shader_program)

        self.bind_texture_3d(0, temperature)
        self.bind_texture_3d(1, density)
        self.bind_texture_3d(2, obstacle)
        self.bind_image_3d_write(0, force_out, GL_RGBA16F)

        glUniform1f(self.get_uniform_loc("uSigma"), sigma)
        glUniform1f(self.get_uniform_loc("uKappa"), kappa)
        glUniform3f(self.get_uniform_loc("uGravity"), gravity[0], gravity[1], gravity[2])
        glUniform3i(self.get_uniform_loc("uSize"),
                    force_out.width, force_out.height, force_out.depth)
        glUniform1i(self.get_uniform_loc("uHasObstacles"), int(has_obstacles))

        self.dispatch(force_out.width, force_out.height, force_out.depth)
