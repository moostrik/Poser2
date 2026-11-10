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
    """The light a composite lights the density by: the light grid, how it is read, and the rays."""
    volume: Texture3D                       # the light reaching each place, RGB, over the box
    intensity: float                        # the lit dye's emission over the whole view at full light (`view_march`'s)
    anisotropy: float                       # Henyey–Greenstein g
    point: bool                             # a lamp at `vector`'s place, else a sun toward `vector`
    vector: tuple[float, float, float]      # ±1 the box's sides
    box: tuple[float, float, float]         # the box in widths
    caustics: float = 0.0                   # the caustic net's contrast, 0 off to 1
    caustic_size: float = 0.1               # the net's cell, widths
    caustic_depth: float = 0.5              # widths down the light's way over which the net fades to 37%
    net: Texture | None = None              # the caustic net, tileable, its mean 1
    rays: Texture3D | None = None           # the water's light per column and layer of the march, None off


def bind_light(shader: ComputeShader, light: CompositeLight, net: Texture | None, unit: int) -> None:
    """The uniforms `light.glsl` reads, the caustic net on texture `unit`."""
    glUniform1i(shader.get_uniform_loc("uPoint"), int(light.point))
    glUniform3f(shader.get_uniform_loc("uVector"), *light.vector)
    glUniform3f(shader.get_uniform_loc("uBox"), *light.box)
    glUniform1f(shader.get_uniform_loc("uAnisotropy"), light.anisotropy)
    on: bool = light.caustics > 0.0 and net is not None and net.allocated
    glUniform1f(shader.get_uniform_loc("uCaustics"), light.caustics if on else 0.0)
    glUniform1f(shader.get_uniform_loc("uCausticSize"), light.caustic_size)
    glUniform1f(shader.get_uniform_loc("uCausticDepth"), light.caustic_depth)
    if on and net is not None:
        shader.bind_texture(unit, net, "uCausticNet")
    else:       # its own unit even unread: a sampler2D on a unit with a sampler3D fails the dispatch
        glUniform1i(shader.get_uniform_loc("uCausticNet"), unit)


class Composite3D(ComputeShader):
    """Composite the density volume into the 2D output: per-channel Beer–Lambert occlusion, the water's
    tint and fog, the light on the dye and in the water. The output may be larger than the volume; the
    volume is sampled by uv."""

    def use(self, density: Texture3D, output_2d: Texture, *,
            emission: float, absorption: float, tint: float, fog: float,
            color: Texture3D | None = None, light: CompositeLight | None = None) -> None:
        """Composite the density volume into the 2D output, one march step per layer.

        Args:
            density: Density volume (RGBA16F)
            output_2d: 2D output texture (RGBA16F)
            emission: The emission of an amount of 1 over the whole view (`view_march`'s)
            absorption: The extinction of an amount of 1 over the whole view (`view_march`'s)
            tint: 0..1 mix toward per-channel water extinction (red dies first)
            fog: Medium extinction over the full depth
            color: With it, `density` is the amount alone and its color is this volume's (color × amount, amount)
            light: With it, each sample also lit (its color × the light there × `light.intensity`) and, with its
                rays, the water's light added a layer at a time
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
        glUniform1i(self.get_uniform_loc("uSplit"), int(color is not None))
        if color is not None:
            self.bind_texture_3d(1, color, "uColor")
        glUniform1i(self.get_uniform_loc("uLit"), int(light is not None))
        rays: Texture3D | None = light.rays if light is not None else None
        glUniform1i(self.get_uniform_loc("uHasRays"), int(rays is not None))
        if light is not None:
            self.bind_texture_3d(2, light.volume, "uLight")
            glUniform1f(self.get_uniform_loc("uIntensity"), light.intensity)
            bind_light(self, light, light.net, 4)
            if rays is not None:
                self.bind_texture_3d(3, rays, "uRays")
        else:       # the net's sampler on its own unit, unread
            glUniform1i(self.get_uniform_loc("uCausticNet"), 4)

        self.dispatch(output_2d.width, output_2d.height)
