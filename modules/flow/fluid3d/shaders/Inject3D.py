"""Inject3D - Inject 2D input into 3D volume at a target depth with gaussian spread."""
from __future__ import annotations

from OpenGL.GL import *  # type: ignore

from modules.gl.Texture import Texture
from modules.gl.Texture3D import Texture3D
from modules.gl.ComputeShader import ComputeShader


class Inject3D(ComputeShader):
    """Inject 2D texture into 3D volume with gaussian depth spread."""

    WORKGROUP_SIZE_X = 16
    WORKGROUP_SIZE_Y = 16
    WORKGROUP_SIZE_Z = 1

    def __init__(self) -> None:
        super().__init__()

    def use(self, input_2d: Texture, volume: Texture3D,
            target_layer: float, spread: float, strength: float = 1.0,
            mode: int = 0, internal_format: int = GL_RGBA16F) -> None:
        """Inject 2D input into 3D volume.

        Args:
            input_2d: 2D source texture
            volume: 3D destination volume (read-write for additive mode)
            target_layer: Normalized depth [0, 1] — center of injection
            spread: Gaussian sigma in normalized depth units
            strength: Injection strength multiplier
            mode: 0 = additive, 1 = set (replace)
            internal_format: Image format for volume binding
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
        glUniform1f(self.get_uniform_loc("uTargetLayer"), target_layer)
        glUniform1f(self.get_uniform_loc("uSpread"), spread)
        glUniform1f(self.get_uniform_loc("uStrength"), strength)
        glUniform1i(self.get_uniform_loc("uMode"), mode)
        glUniform3i(self.get_uniform_loc("uSize"),
                    volume.width, volume.height, volume.depth)

        self.dispatch(volume.width, volume.height, volume.depth)
