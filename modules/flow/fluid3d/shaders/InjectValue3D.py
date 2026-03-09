"""InjectValue3D - Inject brightness (V=max(r,g,b)) from RGBA 2D input into R16F 3D volume."""
from __future__ import annotations

from OpenGL.GL import *  # type: ignore

from modules.gl.Texture import Texture
from modules.gl.Texture3D import Texture3D
from modules.gl.ComputeShader import ComputeShader


class InjectValue3D(ComputeShader):
    """Inject brightness V = max(r,g,b) from a 2D RGBA texture into a R16F 3D volume.

    Uses the same gaussian depth-spread as Inject3D.  Used when the density
    volume stores scalar brightness separate from the colour volume.
    """

    WORKGROUP_SIZE_X = 16
    WORKGROUP_SIZE_Y = 16
    WORKGROUP_SIZE_Z = 1

    def __init__(self) -> None:
        super().__init__()

    def use(self, input_2d: Texture, volume: Texture3D,
            target_layer: float, spread: float, strength: float = 1.0,
            mode: int = 0) -> None:
        """Inject brightness from 2D RGBA input into R16F 3D volume.

        Args:
            input_2d:     2D RGBA source texture.
            volume:       R16F 3D destination volume (read-write for additive).
            target_layer: Normalised depth [0, 1] — centre of injection gaussian.
            spread:       Gaussian sigma in normalised depth units.
            strength:     Injection strength multiplier.
            mode:         0 = additive, 1 = set (replace).
        """
        if not self.allocated or not self.shader_program:
            return

        glUseProgram(self.shader_program)

        # Bind 2D input as sampler
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, input_2d.tex_id)
        glUniform1i(self.get_uniform_loc("uInput"), 0)

        # Bind R16F volume as read-write image
        self.bind_image_3d_readwrite(0, volume, GL_R16F)

        # Uniforms
        glUniform1f(self.get_uniform_loc("uTargetLayer"), target_layer)
        glUniform1f(self.get_uniform_loc("uSpread"), spread)
        glUniform1f(self.get_uniform_loc("uStrength"), strength)
        glUniform1i(self.get_uniform_loc("uMode"), mode)
        glUniform3i(self.get_uniform_loc("uSize"),
                    volume.width, volume.height, volume.depth)

        self.dispatch(volume.width, volume.height, volume.depth)
