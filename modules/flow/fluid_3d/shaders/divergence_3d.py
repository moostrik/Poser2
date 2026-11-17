"""Divergence3D - the divergence of the velocity on the cells' faces, less the volume source."""
from __future__ import annotations

from OpenGL.GL import GL_R16F, glUniform1i, glUseProgram

from modules.gl import StorageBuffer, Texture3D, ComputeShader
from .face_binding import Cell, bind_faces


class Divergence3D(ComputeShader):
    """Per cell, its six faces' one-cell differences over h, less the volume source φ where there is one."""

    WORKGROUP_SIZE_X = 8
    WORKGROUP_SIZE_Y = 8
    WORKGROUP_SIZE_Z = 4

    def use(self, faces: Texture3D, obstacle: Texture3D,
            divergence_out: Texture3D, cell: Cell, has_obstacles: bool = True,
            closed: tuple[bool, bool, bool] = (True, True, True),
            source: tuple[Texture3D, StorageBuffer | None] | None = None) -> None:
        """The velocity's divergence into `divergence_out` (R16F at the cells).

        Args:
            faces: The velocity's faces (`FaceVelocity`)
            obstacle: The obstacle mask (R8 at the cells)
            divergence_out: The divergence, 1/s
            cell: A cell's sides in widths, x, y and z
            has_obstacles: Whether obstacle logic is active
            closed: Per axis, whether the walls are closed
            source: The volume source φ (R16F at the cells, 1/s) and the buffer with its mean to take off (a closed
                box), or None for none; None for no source
        """
        if not self.allocated or not self.shader_program:
            return
        glUseProgram(self.shader_program)
        cells = (divergence_out.width, divergence_out.height, divergence_out.depth)
        bind_faces(self, faces, cells, cell, closed, has_obstacles, obstacle)
        glUniform1i(self.get_uniform_loc('uHasSource'), int(source is not None))
        glUniform1i(self.get_uniform_loc('uSubtractMean'), int(source is not None and source[1] is not None))
        if source is not None:
            volume, mean = source
            self.bind_texture_3d(3, volume, 'uSource')
            if mean is not None:
                mean.bind_base(1)
        self.bind_image_3d_write(0, divergence_out, GL_R16F)
        self.dispatch(*cells)
