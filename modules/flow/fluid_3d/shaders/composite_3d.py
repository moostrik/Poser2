"""Composite3D - the density volume marched front to back into the 2D output."""
from __future__ import annotations

import math
from dataclasses import dataclass

from OpenGL.GL import GL_RGBA16F, glUniform1f, glUniform1i, glUniform2f, glUniform2i, glUniform3f, glUseProgram

from modules.gl import Texture, Texture3D, ComputeShader


IDENTITY: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)
OPACITY_LIMIT: float = 0.999     # opacity 1, covering all, needs infinitely thick dye: it is drawn as this


def henyey_greenstein(g: float, cosine: float) -> float:
    """Henyey–Greenstein's phase, 1 for g = 0, at the cosine between the light's travel and the way to the eye."""
    return (1.0 - g * g) / (1.0 + g * g - 2.0 * g * cosine) ** 1.5


def sun_phase(vector: tuple[float, float, float], anisotropy: float, back_scattering: float) -> float:
    """A sun's phase toward the viewer, the same everywhere: `light.glsl`'s two lobes at the sun's z."""
    if anisotropy <= 0.0:
        return 1.0
    length: float = math.sqrt(sum(c * c for c in vector))
    cosine: float = vector[2] / length if length > 1e-6 else 0.0
    forward: float = henyey_greenstein(anisotropy, cosine)
    return forward + (henyey_greenstein(-anisotropy, cosine) - forward) * back_scattering


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
    """The light a composite lights the density by: the light grid and how it is read, and the light per
    column and layer of the march (`LayerLight3D`) the composite reads."""
    volume: Texture3D                       # the light reaching each place, RGB, over the box; alpha the density's alone
    intensity: float                        # the lit dye's emission over the whole view at full light (`view_march`'s)
    anisotropy: float                       # Henyey–Greenstein g
    back_scattering: float                  # the share of a backward lobe as strong
    point: bool                             # a lamp at `vector`'s place, else a sun toward `vector`
    vector: tuple[float, float, float]      # ±1 the box's sides
    box: tuple[float, float, float]         # the box in widths
    grid: tuple[int, int, int] = (1, 1, 1)  # the light grid's cells, unsheared
    light_map: tuple[float, ...] = IDENTITY # a place's texture coordinates to `volume`'s, row-major 4 × 4
    caustics: float = 0.0                   # the caustic net's contrast, 0 off to 1
    caustic_size: float = 0.1               # the net's cell, widths
    caustic_depth: float = 0.5              # widths down the light's way over which the net fades to 37%
    net: Texture | None = None              # the caustic net, tileable, its mean 1
    direct: Texture3D | None = None         # the light on the dye per layer, the sun's and the sky's; alpha the net's contrast
    rays: Texture3D | None = None           # the water's light per layer, None off


def net_tiles(vector: tuple[float, float, float], box: tuple[float, float, float], size: float) -> tuple[tuple[float, float], ...]:
    """A sun's caustic net laid across its way: the net's tile coordinates per uvw x, y and z (`light.glsl`'s
    uNetX, uNetY, uNetZ), a tile holding 8 cells of `size` widths."""
    length: float = math.sqrt(sum(c * c for c in vector))
    axis = tuple(c / length for c in vector) if length > 1e-6 else (0.0, -1.0, 0.0)
    helper = (0.0, 1.0, 0.0) if abs(axis[1]) < 0.9 else (1.0, 0.0, 0.0)
    side = _cross(axis, helper)
    side_length: float = math.sqrt(sum(c * c for c in side))
    side = tuple(c / side_length for c in side)
    up = _cross(axis, side)
    scale: float = 1.0 / (8.0 * max(size, 1e-4))
    return tuple((box[k] * side[k] * scale, box[k] * up[k] * scale) for k in range(3))


def _cross(a: tuple[float, ...], b: tuple[float, ...]) -> tuple[float, float, float]:
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def bind_light(shader: ComputeShader, light: CompositeLight, net: Texture | None, unit: int) -> None:
    """The uniforms `light.glsl` reads, the caustic net on texture `unit`."""
    glUniform1i(shader.get_uniform_loc("uPoint"), int(light.point))
    glUniform3f(shader.get_uniform_loc("uVector"), *light.vector)
    glUniform3f(shader.get_uniform_loc("uBox"), *light.box)
    glUniform1f(shader.get_uniform_loc("uAnisotropy"), light.anisotropy)
    glUniform1f(shader.get_uniform_loc("uBackScattering"), light.back_scattering)
    glUniform1f(shader.get_uniform_loc("uSunPhase"), sun_phase(light.vector, light.anisotropy, light.back_scattering))
    on: bool = light.caustics > 0.0 and net is not None and net.allocated
    glUniform1f(shader.get_uniform_loc("uCaustics"), light.caustics if on else 0.0)
    glUniform1f(shader.get_uniform_loc("uCausticSize"), light.caustic_size)
    glUniform1f(shader.get_uniform_loc("uCausticDepth"), light.caustic_depth)
    for name, tile in zip(("uNetX", "uNetY", "uNetZ"), net_tiles(light.vector, light.box, light.caustic_size)):
        glUniform2f(shader.get_uniform_loc(name), *tile)
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
            light: With its light per layer, each sample also lit: its color × that light, the caustic net on the
                sun's share, × `light.intensity`; with its rays, the water's light added
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
        direct: Texture3D | None = light.direct if light is not None else None
        rays: Texture3D | None = light.rays if light is not None else None
        glUniform1i(self.get_uniform_loc("uLit"), int(direct is not None))
        glUniform1i(self.get_uniform_loc("uHasRays"), int(rays is not None))
        if light is not None:
            glUniform1f(self.get_uniform_loc("uIntensity"), light.intensity)
            bind_light(self, light, light.net, 4)
        else:       # the net's sampler on its own unit, unread
            glUniform1i(self.get_uniform_loc("uCausticNet"), 4)
        if direct is not None:
            self.bind_texture_3d(2, direct, "uDirect")
        if rays is not None:
            self.bind_texture_3d(3, rays, "uRays")

        self.dispatch(output_2d.width, output_2d.height)
