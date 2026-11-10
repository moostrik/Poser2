"""Rays3D - the light the water sends toward the viewer, per layer of the march, integrated exactly."""
from __future__ import annotations

from OpenGL.GL import GL_RGBA16F, glUniform1f, glUniform1i, glUniform3f, glUniform3i, glUseProgram

from modules.gl import Texture3D, ComputeShader
from .composite_3d import CompositeLight, bind_light


class Rays3D(ComputeShader):
    """The fog's in-scattering of the light, one value a column and a layer of the march."""

    WORKGROUP_SIZE_X = 8
    WORKGROUP_SIZE_Y = 8
    WORKGROUP_SIZE_Z = 4

    def use(self, light: CompositeLight, rays: Texture3D, *, fog: tuple[float, float, float], fog_grey: float,
            scatter: float, sub_steps: int) -> None:
        """The rays into `rays` (RGBA16F, the light grid's columns × the march's layers).

        Args:
            light: The light, as the composite reads it, its caustic net with it
            rays: Written
            fog: The water's extinction over the whole depth, per channel (`fog` × the tint's channels)
            fog_grey: The water's extinction over the whole depth without the tint (`fog`), what it scatters by
            scatter: The fog's share of what it takes out sent on, × the light's intensity
            sub_steps: Sub-steps a layer, at least the light grid's cells a layer
        """
        if not self.allocated or not self.shader_program:
            return
        glUseProgram(self.shader_program)
        self.bind_texture_3d(0, light.volume, "uLight")
        bind_light(self, light, light.net, 1)
        self.bind_image_3d_write(0, rays, GL_RGBA16F)
        glUniform3i(self.get_uniform_loc("uSize"), rays.width, rays.height, rays.depth)
        glUniform3f(self.get_uniform_loc("uFog"), *fog)
        glUniform1f(self.get_uniform_loc("uFogGrey"), fog_grey)
        glUniform1f(self.get_uniform_loc("uScatter"), scatter)
        glUniform1i(self.get_uniform_loc("uSubSteps"), max(1, sub_steps))
        self.dispatch(rays.width, rays.height, rays.depth)
