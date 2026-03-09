"""Inject3DDensity - Inject 2D input into density volume stored as GL_TEXTURE_2D_ARRAY.

Subclass of Inject3D that binds the volume as image2DArray instead of image3D.
"""
from __future__ import annotations
from pathlib import Path

from OpenGL.GL import *  # type: ignore

from modules.gl.Texture import Texture
from modules.gl.Texture2DArray import Texture2DArray
from ...fluid3d.shaders.Inject3D import Inject3D


class Inject3DDensity(Inject3D):
    """Inject 2D texture into density volume (GL_TEXTURE_2D_ARRAY)."""

    def __init__(self) -> None:
        super().__init__()
        self.compute_file_path = Path(__file__).parent / "inject3d_density.comp"

    def use(self, input_2d: Texture, volume: Texture2DArray,
            target_layer: float, spread: float, strength: float = 1.0,
            mode: int = 0, internal_format: int = GL_RGBA16F) -> None:
        """Inject 2D input into density 2D array volume.

        Args:
            input_2d: 2D source texture
            volume: Density volume (Texture2DArray — image2DArray read-write)
            target_layer: Normalized depth [0, 1]
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

        # Bind density volume as image2DArray read-write
        self.bind_image_2d_array_readwrite(0, volume, internal_format)

        # Uniforms
        glUniform1f(self.get_uniform_loc("uTargetLayer"), target_layer)
        glUniform1f(self.get_uniform_loc("uSpread"), spread)
        glUniform1f(self.get_uniform_loc("uStrength"), strength)
        glUniform1i(self.get_uniform_loc("uMode"), mode)
        glUniform3i(self.get_uniform_loc("uSize"),
                    volume.width, volume.height, volume.depth)

        self.dispatch(volume.width, volume.height, volume.depth)
