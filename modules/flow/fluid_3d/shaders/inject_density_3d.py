"""InjectDensity3D - the bridge's density, its amount and its color, blended into the density volume."""
from __future__ import annotations

from OpenGL.GL import GL_RGBA16F, glUniform1f, glUniform1i, glUniform3i, glUseProgram

from modules.gl import Texture, Texture3D, ComputeShader
from .z_position_binding import bind_z_position


class InjectDensity3D(ComputeShader):
    """Blend the color × the amount (`density_blend.glsl`) into the density's slab at each pixel's z position."""

    WORKGROUP_SIZE_X = 16
    WORKGROUP_SIZE_Y = 16
    WORKGROUP_SIZE_Z = 1

    def use(self, amount: Texture, color: Texture, volume: Texture3D, spread: float, strength: float, blend: int,
            gain: float, release: float, oklab: bool, z_position: Texture | None = None) -> None:
        """`amount` the bridge's (red), `color` premultiplied by its alpha; `spread` the slab's sigma in the
        depth's share; `strength` over's and fill's drag share this step, or add's dt; `blend` `DensityBlend`;
        `gain` the amount times this; `release` the release's share where the cell falls (0 = never falls);
        `oklab` the volume holds Oklab; `z_position` (red over green), a pixel without one, or every pixel
        without it, puts nothing in."""
        if not self.allocated or not self.shader_program or not amount.allocated or not color.allocated:
            return
        glUseProgram(self.shader_program)
        amount.bind(0)
        color.bind(2)
        self.bind_image_readwrite(0, volume, GL_RGBA16F)
        bind_z_position(self, 1, z_position)
        glUniform1f(self.get_uniform_loc("uSpread"), spread)
        glUniform1f(self.get_uniform_loc("uStrength"), strength)
        glUniform1i(self.get_uniform_loc("uBlend"), blend)
        glUniform1f(self.get_uniform_loc("uGain"), gain)
        glUniform1f(self.get_uniform_loc("uRelease"), release)
        glUniform1i(self.get_uniform_loc("uOklab"), int(oklab))
        glUniform3i(self.get_uniform_loc("uSize"), volume.width, volume.height, volume.depth)
        self.dispatch(volume.width, volume.height)     # a column each
