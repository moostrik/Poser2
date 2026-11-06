"""ParticleAdvect - the particles one step along the velocity, RK2, faded and aged."""
from __future__ import annotations

import math

from OpenGL.GL import glUniform1f, glUniform1i, glUniform1ui, glUniform3f, glUseProgram

from modules.gl import ComputeShader, StorageBuffer, Texture3D


class ParticleAdvect(ComputeShader):
    """Move every particle one step along the velocity (RK2), fade and evaporate its mass, grow its age."""

    WORKGROUP_SIZE_X = 256
    WORKGROUP_SIZE_Y = 1
    WORKGROUP_SIZE_Z = 1

    def __init__(self) -> None:
        super().__init__()

    def use(self, particles: StorageBuffer, count: int, velocity: Texture3D, *, aspect: float, z_extent: float,
            timestep: float, dt: float, dissipation: float, evaporation: float, closed_xy: bool, closed_z: bool) -> None:
        if not self.allocated or not self.shader_program:
            return
        glUseProgram(self.shader_program)
        particles.bind_base(0)
        self.bind_texture_3d(0, velocity, "uVelocity")
        glUniform1f(self.get_uniform_loc("uTimestep"), timestep)
        glUniform3f(self.get_uniform_loc("uRdx"), 1.0, aspect,
                    1.0 / z_extent if z_extent > 0.0 else 1.0)
        glUniform1f(self.get_uniform_loc("uDt"), dt)
        glUniform1f(self.get_uniform_loc("uDissipation"), dissipation)
        glUniform1f(self.get_uniform_loc("uEvaporation"), evaporation)
        glUniform1i(self.get_uniform_loc("uClosedXY"), int(closed_xy))
        glUniform1i(self.get_uniform_loc("uClosedZ"), int(closed_z))
        glUniform1ui(self.get_uniform_loc("uCount"), count)
        self.dispatch_groups(math.ceil(count / self.WORKGROUP_SIZE_X))
        self.barrier_buffer()
