"""InjectAmount3D - a 2D source's amount into a one-channel volume, in the injection's gaussian slab."""
from __future__ import annotations

from OpenGL.GL import GL_R16F, glUniform1f, glUniform1i, glUniform3i, glUseProgram

from modules.gl import Texture, Texture3D, ComputeShader
from .z_position_binding import bind_z_position


class InjectAmount3D(ComputeShader):
    """Add the source's alpha × strength into the volume's slab at a depth."""

    WORKGROUP_SIZE_X = 16
    WORKGROUP_SIZE_Y = 16
    WORKGROUP_SIZE_Z = 1

    def use(self, source: Texture, volume: Texture3D, target_layer: float, spread: float, strength: float,
            cubic: bool = False, mode: int = 0, gain: float = 1.0, z_position: Texture | None = None) -> None:
        if not self.allocated or not self.shader_program:
            return
        glUseProgram(self.shader_program)
        self.bind_texture(0, source, "uInput")
        self.bind_image_3d_readwrite(0, volume, GL_R16F)
        bind_z_position(self, 1, z_position, target_layer)
        glUniform1f(self.get_uniform_loc("uSpread"), spread)
        glUniform1f(self.get_uniform_loc("uStrength"), strength)
        glUniform1i(self.get_uniform_loc("uCubic"), int(cubic))
        glUniform1i(self.get_uniform_loc("uMode"), mode)
        glUniform1f(self.get_uniform_loc("uGain"), gain)
        glUniform3i(self.get_uniform_loc("uSize"), volume.width, volume.height, volume.depth)
        self.dispatch(volume.width, volume.height)     # a column each
