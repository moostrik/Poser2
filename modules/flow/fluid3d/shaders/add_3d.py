"""Add3D - In-place volumetric addition: dst += src * strength."""
from __future__ import annotations

from OpenGL.GL import *  # type: ignore

from modules.gl import Texture3D, ComputeShader


class Add3D(ComputeShader):
    """In-place 3D volume addition via compute shader.

    Works on identically-sized volumes. No ping-pong needed since each
    thread writes to its own unique voxel (no read-after-write hazards
    within a single dispatch).
    """

    WORKGROUP_SIZE_X = 16
    WORKGROUP_SIZE_Y = 16
    WORKGROUP_SIZE_Z = 1

    def __init__(self) -> None:
        super().__init__()

    def use(self, dst: Texture3D, src: Texture3D,
            strength: float = 1.0,
            dst_format: int = GL_RGBA16F,
            src_format: int = GL_RGBA16F) -> None:
        """Add source volume to destination in-place.

        dst[pos] += src[pos] * strength

        Args:
            dst: Destination 3D texture (read-write)
            src: Source 3D texture (read-only)
            strength: Multiplier for source values
            dst_format: Image format for destination binding
            src_format: Image format for source binding
        """
        if not self.allocated or not self.shader_program:
            return

        glUseProgram(self.shader_program)

        self.bind_image_3d_readwrite(0, dst, dst_format)
        self.bind_image_3d_read(1, src, src_format)

        glUniform1f(self.get_uniform_loc("uStrength"), strength)
        glUniform3i(self.get_uniform_loc("uSize"),
                    dst.width, dst.height, dst.depth)

        self.dispatch(dst.width, dst.height, dst.depth)
