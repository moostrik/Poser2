"""Blit3D - Copy/resample one 3D volume into another at potentially different XY resolution."""
from OpenGL.GL import *  # type: ignore

from modules.gl.Texture3D import Texture3D
from modules.gl.ComputeShader import ComputeShader


class Blit3D(ComputeShader):
    """Copy a 3D volume into another 3D volume, resampling XY as needed.

    Source is bound as a sampler3D so OpenGL handles filtering (nearest
    for binary masks, linear for continuous fields).  Dispatched at the
    destination resolution.
    """

    WORKGROUP_SIZE_X = 16
    WORKGROUP_SIZE_Y = 16
    WORKGROUP_SIZE_Z = 1

    def __init__(self) -> None:
        super().__init__()

    def use(self, source: Texture3D, destination: Texture3D,
            internal_format: int = GL_R8) -> None:
        """Copy source volume into destination volume.

        Args:
            source: 3D volume to read (bound as sampler, uses its filter mode)
            destination: 3D volume to write (bound as image)
            internal_format: Image format for destination binding
        """
        if not self.allocated or not self.shader_program:
            return

        glUseProgram(self.shader_program)

        # Bind source as sampler
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_3D, source.tex_id)
        glUniform1i(self.get_uniform_loc("uSource"), 0)

        # Bind destination as write-only image
        self.bind_image_3d_write(0, destination, internal_format)

        # Destination size (dispatch grid)
        glUniform3i(self.get_uniform_loc("uDstSize"),
                    destination.width, destination.height, destination.depth)

        self.dispatch(destination.width, destination.height, destination.depth)
