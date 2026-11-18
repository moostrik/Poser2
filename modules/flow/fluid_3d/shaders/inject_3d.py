"""Inject3D - Inject a 2D scalar into a 3D volume at each pixel's z position with gaussian spread."""
from __future__ import annotations

from enum import IntEnum

from OpenGL.GL import GL_R16F, GL_TEXTURE0, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1f, glUniform1i, glUniform3i, glUseProgram

from modules.gl import Texture, Texture3D, ComputeShader
from .z_position_binding import bind_z_position


class InjectMode(IntEnum):
    """How a scalar source goes into a field."""
    ADD = 0         # added × dt, unbounded
    FILL = 1        # the field dragged toward the source by the drag's share, never past it, only where the source is further from 0


class Inject3D(ComputeShader):
    """Add a 2D scalar (red, signed) into a 3D volume with gaussian depth spread; the density goes in by
    `InjectDensity3D`."""

    WORKGROUP_SIZE_X = 16
    WORKGROUP_SIZE_Y = 16
    WORKGROUP_SIZE_Z = 1

    def __init__(self) -> None:
        super().__init__()

    def use(self, input_2d: Texture, volume: Texture3D,
            spread: float, strength: float = 1.0,
            internal_format: int = GL_R16F, mode: int = 0, gain: float = 1.0,
            z_position: Texture | None = None) -> None:
        """Add the 2D input into the 3D volume.

        Args:
            input_2d: 2D source texture, the scalar in red
            volume: 3D destination volume (read-write)
            spread: Gaussian sigma in normalized depth units
            strength: Injection strength multiplier
            internal_format: Image format for volume binding
            mode: 0 ADD, 1 FILL (dragged toward the input, never past it, only where it is further from 0)
            gain: FILL: the input times this, at most ±1, is the target
            z_position: The input's z position (red over green); a pixel without one, or every pixel without
                it, puts nothing in
        """
        if not self.allocated or not self.shader_program:
            return

        glUseProgram(self.shader_program)

        # Bind 2D input as sampler
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, input_2d.tex_id)
        glUniform1i(self.get_uniform_loc("uInput"), 0)

        # Bind 3D volume as read-write image
        self.bind_image_3d_readwrite(0, volume, internal_format)

        # Uniforms
        bind_z_position(self, 1, z_position)
        glUniform1f(self.get_uniform_loc("uSpread"), spread)
        glUniform1f(self.get_uniform_loc("uStrength"), strength)
        glUniform1i(self.get_uniform_loc("uMode"), mode)
        glUniform1f(self.get_uniform_loc("uGain"), gain)
        glUniform3i(self.get_uniform_loc("uSize"),
                    volume.width, volume.height, volume.depth)

        self.dispatch(volume.width, volume.height)     # a column each
