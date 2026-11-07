"""NoiseCoordinates3D - the water's displacement since the last reset, so the turbulence's noise travels with it."""
from __future__ import annotations

from OpenGL.GL import GL_RGBA16F, glUniform1f, glUniform1i, glUniform3f, glUniform3i, glUseProgram

from modules.gl import Texture3D, ComputeShader


class NoiseCoordinates3D(ComputeShader):
    """d(x) = d(x - v dt) + v dt; or 0 at a reset."""

    WORKGROUP_SIZE_X = 8
    WORKGROUP_SIZE_Y = 8
    WORKGROUP_SIZE_Z = 4

    def use(self, velocity: Texture3D, source: Texture3D, result: Texture3D, timestep: float,
            rdx: tuple[float, float, float], reset: bool) -> None:
        if not self.allocated or not self.shader_program:
            return
        glUseProgram(self.shader_program)
        self.bind_texture_3d(0, velocity, "uVelocity")
        self.bind_texture_3d(1, source, "uSource")
        self.bind_image_3d_write(0, result, GL_RGBA16F)
        glUniform1f(self.get_uniform_loc("uTimestep"), timestep)
        glUniform3f(self.get_uniform_loc("uRdx"), *rdx)
        glUniform1i(self.get_uniform_loc("uReset"), int(reset))
        glUniform3i(self.get_uniform_loc("uSize"), result.width, result.height, result.depth)
        self.dispatch(result.width, result.height, result.depth)
