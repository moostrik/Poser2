"""InjectBinary3D - Project 2D binary mask into all layers of a 3D R8 volume."""
from __future__ import annotations

from OpenGL.GL import *  # type: ignore

from modules.gl import Texture, Texture3D, ComputeShader


class InjectBinary3D(ComputeShader):
    """Project a 2D binary mask uniformly into every layer of a 3D R8 volume.

    Used for obstacle injection — a 2D obstacle applies at every depth.
    Input values are binarised at 0.5.

    Modes:
        0 = OR  (boolean union with existing data)
        1 = SET (replace volume with the mask)
    """

    WORKGROUP_SIZE_X = 16
    WORKGROUP_SIZE_Y = 16
    WORKGROUP_SIZE_Z = 1

    def __init__(self) -> None:
        super().__init__()

    def use(self, input_2d: Texture, volume: Texture3D, mode: int = 0) -> None:
        """Inject 2D binary mask into 3D volume.

        Args:
            input_2d: 2D source texture (reads .r, binarised at 0.5)
            volume: 3D R8 destination volume (read-write)
            mode: 0 = OR (boolean union), 1 = SET (replace)
        """
        if not self.allocated or not self.shader_program:
            return

        glUseProgram(self.shader_program)

        # Bind 2D input as sampler
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, input_2d.tex_id)
        glUniform1i(self.get_uniform_loc("uInput"), 0)

        # Bind 3D volume as read-write image (R8 format)
        self.bind_image_3d_readwrite(0, volume, GL_R8)

        # Uniforms
        glUniform3i(self.get_uniform_loc("uSize"),
                    volume.width, volume.height, volume.depth)
        glUniform1i(self.get_uniform_loc("uMode"), mode)

        self.dispatch(volume.width, volume.height, volume.depth)
