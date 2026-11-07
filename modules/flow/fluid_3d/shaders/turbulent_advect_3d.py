"""TurbulentAdvect3D - the output-resolution density's step by the grid's velocity plus turbulence below the grid."""
from __future__ import annotations

from OpenGL.GL import GL_R16F, glUniform1f, glUniform1i, glUniform2f, glUniform3f, glUniform3i, glUseProgram

from modules.gl import Texture3D, ComputeShader


class TurbulentAdvect3D(ComputeShader):
    """Semi-Lagrangian, the velocity the grid's plus the noise's swirls scaled by the grid's churn."""

    WORKGROUP_SIZE_X = 8
    WORKGROUP_SIZE_Y = 8
    WORKGROUP_SIZE_Z = 4

    def use(self, source: Texture3D, result: Texture3D, velocity: Texture3D, obstacle: Texture3D, curl: Texture3D,
            noise: Texture3D, displacement_a: Texture3D, displacement_b: Texture3D, *, timestep: float,
            rdx: tuple[float, float, float], extent: tuple[float, float, float], dissipation: float, evaporation: float,
            turbulence: float, cell: float, tiles: float, weights: tuple[float, float], has_obstacles: bool) -> None:
        if not self.allocated or not self.shader_program:
            return
        glUseProgram(self.shader_program)
        self.bind_texture_3d(0, velocity, "uVelocity")
        self.bind_texture_3d(1, source, "uSource")
        self.bind_texture_3d(2, obstacle, "uObstacle")
        self.bind_texture_3d(3, curl, "uCurl")
        self.bind_texture_3d(4, noise, "uNoise")
        self.bind_texture_3d(5, displacement_a, "uDisplacementA")
        self.bind_texture_3d(6, displacement_b, "uDisplacementB")
        self.bind_image_3d_write(0, result, GL_R16F)
        glUniform1f(self.get_uniform_loc("uTimestep"), timestep)
        glUniform3f(self.get_uniform_loc("uRdx"), *rdx)
        glUniform3f(self.get_uniform_loc("uExtent"), *extent)
        glUniform1f(self.get_uniform_loc("uDissipation"), dissipation)
        glUniform1f(self.get_uniform_loc("uEvaporation"), evaporation)
        glUniform1f(self.get_uniform_loc("uTurbulence"), turbulence)
        glUniform1f(self.get_uniform_loc("uCell"), cell)
        glUniform1f(self.get_uniform_loc("uTiles"), tiles)
        glUniform2f(self.get_uniform_loc("uWeights"), *weights)
        glUniform3i(self.get_uniform_loc("uSize"), result.width, result.height, result.depth)
        glUniform1i(self.get_uniform_loc("uHasObstacles"), int(has_obstacles))
        self.dispatch(result.width, result.height, result.depth)
