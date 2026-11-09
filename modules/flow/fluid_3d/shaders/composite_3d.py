"""Composite3D - the density volume marched front to back into the 2D output."""
from __future__ import annotations

import math
from dataclasses import dataclass

from OpenGL.GL import GL_RGBA16F, glUniform1f, glUniform1i, glUniform2i, glUniform3f, glUseProgram

from modules.gl import Texture, Texture3D, ComputeShader


OPACITY_LIMIT: float = 0.999     # opacity 1, covering all, needs infinitely thick dye: it is drawn as this


def view_march(opacity: float, emission: float) -> tuple[float, float]:
    """The march's extinction and emission over the whole view from what the panel says: fully dyed water
    filling the view covers `opacity` of what is behind it (Beer-Lambert: its extinction -ln(1 - opacity)) and
    is seen at its color times `emission`, its own covering of its far side made up for (its emission over
    the view the extinction / the opacity times `emission`; at opacity 0, `emission` itself)."""
    covered: float = min(max(opacity, 0.0), OPACITY_LIMIT)
    extinction: float = -math.log(1.0 - covered)
    return extinction, emission * (extinction / covered if covered > 0.0 else 1.0)


@dataclass
class CompositeLight:
    """The light a composite darkens the density by: the light grid and how it is read."""
    volume: Texture3D                       # RGB transmittance toward the light, over the box
    shadow: float                           # how dark the shadowed parts get, 0 to 1
    anisotropy: float                       # Henyey–Greenstein g
    point: bool                             # a lamp at `vector`'s place, else a sun toward `vector`
    vector: tuple[float, float, float]      # ±1 the box's sides
    box: tuple[float, float, float]         # the box in widths


class Composite3D(ComputeShader):
    """Composite the density volume into the 2D output: per-channel Beer–Lambert occlusion, a glow mix,
    the water's tint and fog. The output may be larger than the volume; the volume is sampled by uv."""

    def use(self, density: Texture3D, output_2d: Texture, *,
            emission: float, absorption: float, tint: float, fog: float, glow: float,
            color: Texture3D | None = None, light: CompositeLight | None = None) -> None:
        """Composite the density volume into the 2D output, one march step per layer.

        Args:
            density: Density volume (RGBA16F)
            output_2d: 2D output texture (RGBA16F)
            emission: The emission of an amount of 1 over the whole view (`view_march`'s)
            absorption: The extinction of an amount of 1 over the whole view (`view_march`'s)
            tint: 0..1 mix toward per-channel water extinction (red dies first)
            fog: Medium extinction over the full depth
            glow: 0 = occluded volumetric, 1 = additive
            color: With it, `density` is the amount alone and its color is this volume's (color × amount, amount)
            light: With it, each sample's emission darkened by the light reaching it, by `light.shadow`
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
        glUniform1i(self.get_uniform_loc("uLit"), int(light is not None))
        if light is not None:
            self.bind_texture_3d(2, light.volume, "uLight")
            glUniform1f(self.get_uniform_loc("uShadow"), light.shadow)
            glUniform1f(self.get_uniform_loc("uAnisotropy"), light.anisotropy)
            glUniform1i(self.get_uniform_loc("uPoint"), int(light.point))
            glUniform3f(self.get_uniform_loc("uVector"), *light.vector)
            glUniform3f(self.get_uniform_loc("uBox"), *light.box)

        self.dispatch(output_2d.width, output_2d.height)
