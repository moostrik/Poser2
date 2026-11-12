"""LightVolume3D - how much light reaches each cell of a coarse grid, through the density and the water."""
from __future__ import annotations

from OpenGL.GL import GL_RGBA16F, glUniform1f, glUniform1i, glUniform3f, glUniform3i, glUseProgram

from modules.gl import Texture3D, ComputeShader


class LightVolume3D(ComputeShader):
    """Each light cell's transmittance toward the light: the density's amount on the way and the water's fog."""

    WORKGROUP_SIZE_X = 8
    WORKGROUP_SIZE_Y = 8
    WORKGROUP_SIZE_Z = 4

    def use(self, density: Texture3D, light: Texture3D, *, box: tuple[float, float, float], point: bool,
            vector: tuple[float, float, float], sigma: tuple[float, float, float],
            fog: tuple[float, float, float]) -> None:
        """March each light cell toward the light.

        Args:
            density: The density (RGBA16F, the amount as alpha)
            light: The light grid (RGBA16F), written
            box: The box in widths: 1, height over width, depth over width
            point: A lamp at `vector`'s place, else a sun toward `vector`
            vector: Toward the sun, or the lamp's place, ±1 the box's sides
            sigma: The density's extinction per width per unit of amount, per channel
            fog: The water's extinction per width, per channel
        """
        if not self.allocated or not self.shader_program:
            return
        glUseProgram(self.shader_program)
        self.bind_texture_3d(0, density, "uDensity")
        self.bind_image_3d_write(0, light, GL_RGBA16F)
        glUniform3i(self.get_uniform_loc("uSize"), light.width, light.height, light.depth)
        glUniform3f(self.get_uniform_loc("uBox"), *box)
        glUniform1i(self.get_uniform_loc("uPoint"), int(point))
        glUniform3f(self.get_uniform_loc("uVector"), *vector)
        glUniform3f(self.get_uniform_loc("uSigma"), *sigma)
        glUniform3f(self.get_uniform_loc("uFog"), *fog)
        glUniform1f(self.get_uniform_loc("uStep"), box[0] / max(light.width, 1))
        self.dispatch(light.width, light.height, light.depth)
