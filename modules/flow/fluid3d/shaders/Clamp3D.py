"""Clamp3D - Clamp all voxels of a 3D volume to a specified range."""
from __future__ import annotations

from OpenGL.GL import *  # type: ignore

from modules.gl.Texture3D import Texture3D
from modules.gl.ComputeShader import ComputeShader


class Clamp3D(ComputeShader):
    """Clamp all voxels of a 3D volume to [min_val, max_val] in-place."""

    WORKGROUP_SIZE_X = 16
    WORKGROUP_SIZE_Y = 16
    WORKGROUP_SIZE_Z = 1

    def __init__(self) -> None:
        super().__init__()

    def use(self, volume: Texture3D, min_val: float = 0.0,
            max_val: float = 1.0, internal_format: int = GL_RGBA16F) -> None:
        """Clamp all voxels of a 3D volume.

        Args:
            volume: 3D volume to clamp (read-write, in-place)
            min_val: Minimum clamp value (applied to all channels)
            max_val: Maximum clamp value (applied to all channels)
            internal_format: Image format for volume binding
        """
        if not self.allocated or not self.shader_program:
            return

        glUseProgram(self.shader_program)

        # Bind 3D volume as read-write image
        self.bind_image_3d_readwrite(0, volume, internal_format)

        # Uniforms
        glUniform1f(self.get_uniform_loc("uMin"), min_val)
        glUniform1f(self.get_uniform_loc("uMax"), max_val)
        glUniform3i(self.get_uniform_loc("uSize"),
                    volume.width, volume.height, volume.depth)

        self.dispatch(volume.width, volume.height, volume.depth)
