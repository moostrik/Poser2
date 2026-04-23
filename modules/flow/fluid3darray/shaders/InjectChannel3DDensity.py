"""InjectChannel3DDensity - Inject single-channel 2D input into density as GL_TEXTURE_2D_ARRAY.

Subclass of InjectChannel3D that binds the volume as image2DArray instead of image3D.
"""
from __future__ import annotations
from pathlib import Path

from OpenGL.GL import *  # type: ignore

from modules.gl import Texture, Texture2DArray
from ...fluid3d.shaders.InjectChannel3D import InjectChannel3D


class InjectChannel3DDensity(InjectChannel3D):
    """Inject single-channel 2D texture into one RGBA channel of density (GL_TEXTURE_2D_ARRAY)."""

    def __init__(self) -> None:
        super().__init__()
        self.compute_file_path = Path(__file__).parent / "injectchannel3d_density.comp"

    def use(self, input_2d: Texture, volume: Texture2DArray,
            target_layer: float, spread: float, channel: int,
            strength: float = 1.0, mode: int = 0,
            internal_format: int = GL_RGBA16F) -> None:
        """Inject single-channel 2D input into one channel of density 2D array.

        Args:
            input_2d: 2D source texture (reads .r component)
            volume: Density volume (Texture2DArray — image2DArray read-write)
            target_layer: Normalized depth [0, 1]
            spread: Gaussian sigma in normalized depth units
            channel: Target RGBA channel (0=R, 1=G, 2=B, 3=A)
            strength: Injection strength multiplier
            mode: 0 = additive, 1 = set
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
        glUniform1i(self.get_uniform_loc("uChannel"), channel)
        glUniform1i(self.get_uniform_loc("uMode"), mode)
        glUniform3i(self.get_uniform_loc("uSize"),
                    volume.width, volume.height, volume.depth)

        self.dispatch(volume.width, volume.height, volume.depth)
