"""ExpansionMean3D - the expansion's mean over the water's cells, on the GPU, for a closed box."""
from __future__ import annotations

import math

from OpenGL.GL import GL_SHADER_STORAGE_BARRIER_BIT, glMemoryBarrier, glUniform1i, glUniform3i, glUseProgram

from modules.gl import ComputeShader, StorageBuffer, Texture3D

PARTIAL_BYTES: int = 8      # a group's partial: the sum and the water cells, two floats


class ExpansionMean3D(ComputeShader):
    """Sums the expansion per workgroup into `partials`, then the partials into `mean`, the sum over the water
    cells; nothing is read back."""

    WORKGROUP_SIZE_X = 8
    WORKGROUP_SIZE_Y = 8
    WORKGROUP_SIZE_Z = 4

    @classmethod
    def groups(cls, cells: tuple[int, int, int]) -> int:
        """The workgroups pass 0 runs over `cells`, the partials it writes."""
        return (math.ceil(cells[0] / cls.WORKGROUP_SIZE_X) * math.ceil(cells[1] / cls.WORKGROUP_SIZE_Y)
                * math.ceil(cells[2] / cls.WORKGROUP_SIZE_Z))

    def use(self, expansion: Texture3D, obstacle: Texture3D, has_obstacles: bool,
            partials: StorageBuffer, mean: StorageBuffer) -> None:
        """The expansion's mean into `mean` (one float).

        Args:
            expansion: The expansion at the cells (R16F)
            obstacle: The obstacle mask (R8 at the cells)
            has_obstacles: Whether an obstacle's cells count for nothing
            partials: At least `groups(cells)` × PARTIAL_BYTES bytes
            mean: Four bytes, the mean
        """
        if not self.allocated or not self.shader_program:
            return
        cells: tuple[int, int, int] = (expansion.width, expansion.height, expansion.depth)
        glUseProgram(self.shader_program)
        expansion.bind(0)
        obstacle.bind(2)
        glUniform3i(self.get_uniform_loc('uCells'), *cells)
        glUniform1i(self.get_uniform_loc('uHasObstacles'), int(has_obstacles))
        partials.bind_base(0)
        mean.bind_base(1)
        glUniform1i(self.get_uniform_loc('uPass'), 0)
        self.dispatch(*cells)
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)
        glUniform1i(self.get_uniform_loc('uPass'), 1)
        glUniform1i(self.get_uniform_loc('uGroups'), self.groups(cells))
        self.dispatch_groups(1)
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)
