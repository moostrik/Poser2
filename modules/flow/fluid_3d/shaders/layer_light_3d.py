"""LayerLight3D - the light per column of the light grid and layer of the march: the light on the dye, the sun's
and the sky's, and the water's own (the rays), each once a light cell and a layer instead of at every 4K sample."""
from __future__ import annotations

import numpy as np
from OpenGL.GL import GL_RGBA16F, GL_TRUE, glUniform1f, glUniform1i, glUniform3f, glUniform3i, glUniformMatrix4fv, glUseProgram

from modules.gl import Texture3D, ComputeShader
from .composite_3d import CompositeLight, bind_light


class LayerLight3D(ComputeShader):
    """Writes the light on the dye (its caustic contrast as alpha) and the rays, a column and a layer each."""

    WORKGROUP_SIZE_X = 8
    WORKGROUP_SIZE_Y = 8
    WORKGROUP_SIZE_Z = 4

    def use(self, light: CompositeLight, sky: Texture3D | None, sky_map: tuple[float, ...], direct: Texture3D,
            rays: Texture3D | None, *, octaves: float, powder: float, sky_fill: float,
            fog: tuple[float, float, float], fog_grey: float, scatter: float, sub_steps: int) -> None:
        """The light per layer into `direct`, and with it `rays` (both RGBA16F, the light grid's columns × the
        march's layers).

        Args:
            light: The light grid and how it is read, its caustic net with it
            sky: The light from straight above, with `sky_fill`
            sky_map: A place's texture coordinates to the sky's, row-major 4 × 4 (it is swept, so stored sheared)
            direct: Written: the sun's light, its phase, octaves and powder in, and `sky_fill` × the sky's; alpha the
                caustic contrast on the sun's share
            rays: Written when given: the water's own light
            octaves: The octaves' share; 0 off
            powder: The powder, 0 off to 1
            sky_fill: The sky's light, a share of the light's intensity
            fog: The water's extinction over the whole depth, per channel (`fog` × the tint's channels)
            fog_grey: The water's extinction over the whole depth without the tint (`fog`), what it scatters by
            scatter: The fog's share of what it takes out sent on, × the light's intensity
            sub_steps: The rays' sub-steps a layer, at least two a light cell
        """
        if not self.allocated or not self.shader_program:
            return
        glUseProgram(self.shader_program)
        self.bind_texture_3d(0, light.volume, "uLight")
        glUniformMatrix4fv(self.get_uniform_loc("uLightMap"), 1, GL_TRUE, np.asarray(light.light_map, np.float32))
        glUniformMatrix4fv(self.get_uniform_loc("uSkyMap"), 1, GL_TRUE, np.asarray(sky_map, np.float32))
        bind_light(self, light, light.net, 1)
        has_sky: bool = sky is not None and sky_fill > 0.0
        if has_sky and sky is not None:
            self.bind_texture_3d(2, sky, "uSky")
        self.bind_image_3d_write(0, direct, GL_RGBA16F)
        if rays is not None:
            self.bind_image_3d_write(2, rays, GL_RGBA16F)
        glUniform3i(self.get_uniform_loc("uSize"), direct.width, direct.height, direct.depth)
        glUniform1f(self.get_uniform_loc("uOctaves"), octaves)
        glUniform1f(self.get_uniform_loc("uPowder"), powder)
        glUniform1f(self.get_uniform_loc("uSkyFill"), sky_fill if has_sky else 0.0)
        glUniform1i(self.get_uniform_loc("uHasRays"), int(rays is not None))
        glUniform3f(self.get_uniform_loc("uFog"), *fog)
        glUniform1f(self.get_uniform_loc("uFogGrey"), fog_grey)
        glUniform1f(self.get_uniform_loc("uScatter"), scatter)
        glUniform1i(self.get_uniform_loc("uSubSteps"), max(1, sub_steps))
        self.dispatch(direct.width, direct.height, direct.depth)
