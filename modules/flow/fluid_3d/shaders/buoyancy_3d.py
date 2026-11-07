"""Buoyancy3D - the buoyancy force: the density's weight along gravity, the temperature's lift against it."""
from __future__ import annotations

from OpenGL.GL import GL_RGBA16F, glUniform1f, glUniform1i, glUniform3f, glUniform3i, glUseProgram

from modules.gl import Texture3D, ComputeShader


class Buoyancy3D(ComputeShader):
    """Compute the 3D buoyancy force: F = gravity * (κ·density − σ(T − T_ambient))."""

    WORKGROUP_SIZE_X = 8
    WORKGROUP_SIZE_Y = 8
    WORKGROUP_SIZE_Z = 4

    def __init__(self) -> None:
        super().__init__()

    def use(self, temperature: Texture3D, density: Texture3D,
            obstacle: Texture3D, force_out: Texture3D,
            sigma: float, kappa: float, ambient_temperature: float,
            gravity: tuple[float, float, float] = (0.0, 1.0, 0.0),
            has_obstacles: bool = True) -> None:
        """Compute the buoyancy force.

        Args:
            temperature: 3D temperature field (R16F)
            density: 3D density field (RGBA16F; its weight is the max of r, g, b)
            obstacle: 3D obstacle mask (R8)
            force_out: Output force volume (RGBA16F)
            sigma: Thermal buoyancy coefficient (includes dt * scale)
            kappa: Density weight coefficient (includes dt * scale)
            ambient_temperature: Reference temperature
            gravity: Gravity direction — (0,1,0) down-screen, (0,0,1) away from viewer
            has_obstacles: Whether obstacle logic is active
        """
        if not self.allocated or not self.shader_program:
            return

        glUseProgram(self.shader_program)

        self.bind_texture_3d(0, temperature, "uTemperature")
        self.bind_texture_3d(1, density, "uDensity")
        self.bind_texture_3d(2, obstacle, "uObstacle")
        self.bind_image_3d_write(0, force_out, GL_RGBA16F)

        glUniform1f(self.get_uniform_loc("uSigma"), sigma)
        glUniform1f(self.get_uniform_loc("uKappa"), kappa)
        glUniform1f(self.get_uniform_loc("uAmbientTemperature"), ambient_temperature)
        glUniform3f(self.get_uniform_loc("uGravity"), gravity[0], gravity[1], gravity[2])
        glUniform3i(self.get_uniform_loc("uSize"),
                    force_out.width, force_out.height, force_out.depth)
        glUniform1i(self.get_uniform_loc("uHasObstacles"), int(has_obstacles))

        self.dispatch(force_out.width, force_out.height, force_out.depth)
