"""Composite3D - the density volume marched front to back into the 2D output."""
from __future__ import annotations

from OpenGL.GL import GL_RGBA16F, glUniform1f, glUniform1i, glUniform2i, glUseProgram

from modules.gl import Texture, Texture3D, ComputeShader


class Composite3D(ComputeShader):
    """Composite the density volume into the 2D output: per-channel Beer–Lambert occlusion, a glow mix,
    the water's tint and fog. The output may be larger than the volume; the volume is sampled by uv."""

    def use(self, density: Texture3D, output_2d: Texture, *,
            emission: float, absorption: float, tint: float, fog: float, glow: float,
            color: Texture3D | None = None) -> None:
        """Composite the density volume into the 2D output, one march step per layer.

        Args:
            density: Density volume (RGBA16F)
            output_2d: 2D output texture (RGBA16F)
            emission: Density brightness gain
            absorption: Extinction per unit density
            tint: 0..1 mix toward per-channel water extinction (red dies first)
            fog: Medium extinction over the full depth
            glow: 0 = occluded volumetric, 1 = additive
            color: With it, `density` is the amount alone and its color is this volume's (color × amount, amount)
        """
        if not self.allocated or not self.shader_program:
            return

        glUseProgram(self.shader_program)

        self.bind_texture_3d(0, density, "uDensity")
        self.bind_image_write(0, output_2d, GL_RGBA16F)

        glUniform1i(self.get_uniform_loc("uDepth"), density.depth)
        glUniform2i(self.get_uniform_loc("uOutputSize"), output_2d.width, output_2d.height)
        glUniform1f(self.get_uniform_loc("uEmission"), emission)
        glUniform1f(self.get_uniform_loc("uAbsorption"), absorption)
        glUniform1f(self.get_uniform_loc("uTint"), tint)
        glUniform1f(self.get_uniform_loc("uFog"), fog)
        glUniform1f(self.get_uniform_loc("uGlow"), glow)
        glUniform1i(self.get_uniform_loc("uSplit"), int(color is not None))
        if color is not None:
            self.bind_texture_3d(1, color, "uColor")

        self.dispatch(output_2d.width, output_2d.height)
