"""Dampen3D - Exponential drag on magnitude excess in a 3D volume."""
from __future__ import annotations

from OpenGL.GL import *  # type: ignore

from modules.gl.Texture3D import Texture3D
from modules.gl.ComputeShader import ComputeShader


class Dampen3D(ComputeShader):
    """Dampen voxels above a magnitude threshold via exponential drag (in-place)."""

    WORKGROUP_SIZE_X = 16
    WORKGROUP_SIZE_Y = 16
    WORKGROUP_SIZE_Z = 1

    def __init__(self) -> None:
        super().__init__()

    def use(self, volume: Texture3D, threshold: float, dampen_factor: float,
            include_alpha: bool = False,
            internal_format: int = GL_RGBA16F) -> None:
        """Dampen voxels above a magnitude threshold.

        Args:
            volume: 3D volume to dampen (read-write, in-place)
            threshold: Magnitude below which values are untouched
            dampen_factor: Precomputed decay multiplier for excess
                           (pow(0.01, dt / dampen_time); 1.0 = no effect)
            include_alpha: If True, magnitude from RGBA (density).
                           If False, magnitude from RGB only (velocity/temp).
            internal_format: Image format for volume binding
        """
        if not self.allocated or not self.shader_program:
            return

        glUseProgram(self.shader_program)

        # Bind 3D volume as read-write image
        self.bind_image_3d_readwrite(0, volume, internal_format)

        # Uniforms
        glUniform1f(self.get_uniform_loc("uThreshold"), threshold)
        glUniform1f(self.get_uniform_loc("uDampenFactor"), dampen_factor)
        glUniform1i(self.get_uniform_loc("uIncludeAlpha"), int(include_alpha))
        glUniform3i(self.get_uniform_loc("uSize"),
                    volume.width, volume.height, volume.depth)

        self.dispatch(volume.width, volume.height, volume.depth)
