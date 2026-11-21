"""Divergence3D - the divergence of the velocity on the cells' faces, less the expansion's."""
from __future__ import annotations

from OpenGL.GL import GL_R16F, glUniform1f, glUniform1i, glUseProgram

from modules.gl import StorageBuffer, Texture3D, ComputeShader
from .face_binding import Cell, bind_cell, bind_faces, bind_obstacle


class Divergence3D(ComputeShader):
    """Per cell, its six faces' one-cell differences over h, less the expansion's φ where there is one."""

    WORKGROUP_SIZE_X = 8
    WORKGROUP_SIZE_Y = 8
    WORKGROUP_SIZE_Z = 4

    def use(self, faces: Texture3D, obstacle: Texture3D,
            divergence_out: Texture3D, cell: Cell, has_obstacles: bool = True,
            closed: tuple[bool, bool, bool] = (True, True, True),
            expansion: tuple[Texture3D, float, StorageBuffer | None] | None = None) -> None:
        """The velocity's divergence into `divergence_out` (R16F at the cells).

        Args:
            faces: The velocity's faces (`FaceVelocity`)
            obstacle: The obstacle mask (R8 at the cells)
            divergence_out: The divergence, 1/s
            cell: A cell's sides in widths, x, y and z
            has_obstacles: Whether obstacle logic is active
            closed: Per axis, whether the walls are closed
            expansion: The expansion (R16F at the cells), its rate (1/s at full presence) and the buffer with its
                mean to take off (a closed box), or None for none; None for no expansion
        """
        if not self.allocated or not self.shader_program:
            return
        glUseProgram(self.shader_program)
        cells = (divergence_out.width, divergence_out.height, divergence_out.depth)
        bind_faces(self, faces, cells)
        bind_cell(self, cell)
        bind_obstacle(self, obstacle, has_obstacles)
        glUniform1i(self.get_uniform_loc('uHasExpansion'), int(expansion is not None))
        glUniform1i(self.get_uniform_loc('uSubtractMean'), int(expansion is not None and expansion[2] is not None))
        if expansion is not None:
            volume, rate, mean = expansion
            self.bind_texture_3d(3, volume, 'uExpansion')
            glUniform1f(self.get_uniform_loc('uExpansionRate'), rate)
            if mean is not None:
                mean.bind_base(1)
        self.bind_image_3d_write(0, divergence_out, GL_R16F)
        self.dispatch(*cells)
