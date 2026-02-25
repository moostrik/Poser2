"""Buoyancy3D - 3D temperature-driven buoyancy force."""
from __future__ import annotations

from OpenGL.GL import *  # type: ignore

from modules.gl.Texture3D import Texture3D
from modules.gl.ComputeShader import ComputeShader


class Buoyancy3D(ComputeShader):
    """Compute 3D buoyancy force: F = σ(T - T_ambient) - κ·density."""

    WORKGROUP_SIZE_X = 16
    WORKGROUP_SIZE_Y = 16
    WORKGROUP_SIZE_Z = 1

    def __init__(self) -> None:
        super().__init__()

    def use(self, temperature: Texture3D, density: Texture3D,
            obstacle: Texture3D, force_out: Texture3D,
            sigma: float, kappa: float, ambient_temperature: float,
            has_obstacles: bool = True) -> None:
        """Compute buoyancy force.

        Args:
            temperature: 3D temperature field (R16F)
            density: 3D density field (RGBA16F)
            obstacle: 3D obstacle mask (R8)
            force_out: Output force volume (RGBA16F)
            sigma: Thermal buoyancy coefficient (includes dt * scale)
            kappa: Density weight coefficient (includes dt * scale)
            ambient_temperature: Reference temperature
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
        glUniform3i(self.get_uniform_loc("uSize"),
                    force_out.width, force_out.height, force_out.depth)
        glUniform1i(self.get_uniform_loc("uHasObstacles"), int(has_obstacles))

        self.dispatch(force_out.width, force_out.height, force_out.depth)
