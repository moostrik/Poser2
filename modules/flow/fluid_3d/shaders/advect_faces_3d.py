"""AdvectFaces3D - the velocity on the cells' faces carried one step along itself."""
from __future__ import annotations

from OpenGL.GL import glUniform1f, glUniform1i, glUseProgram

from modules.gl import Texture3D, ComputeShader
from ...finite_guard import GuardSlot
from .face_binding import Cell, bind_faces, bind_face_writes

SEMI_LAGRANGIAN: int = 0
PREDICTOR: int = 1
CORRECTION: int = 2

class AdvectFaces3D(ComputeShader):
    """Each face traced back on its own; semi-Lagrangian, or MacCormack's predictor and correction."""

    WORKGROUP_SIZE_X = 8
    WORKGROUP_SIZE_Y = 8
    WORKGROUP_SIZE_Z = 4

    def use(self, mode: int, source: Texture3D, result: Texture3D, obstacle: Texture3D,
            cells: tuple[int, int, int], cell: Cell, step: float, dissipation: float = 1.0, evaporation: float = 0.0,
            guard: GuardSlot | None = None, closed: tuple[bool, bool, bool] = (True, True, True),
            wall_keep: float = 1.0, has_obstacles: bool = True, predictor: Texture3D | None = None, clamp: bool = True) -> None:
        """Advect `source` into `result`.

        Args:
            mode: SEMI_LAGRANGIAN, PREDICTOR (the plain step alone) or CORRECTION (MacCormack's, `predictor` read)
            source: The faces before the step (`FaceVelocity`)
            result: The faces written
            cells: The cell counts (width, height, depth)
            cell: A cell's sides in widths, x, y and z
            step: dt × advection_scale
            dissipation: The fade's multiplier this step
            evaporation: Widths/s taken off the speed this step
            guard: The velocity's slot in the stage's FiniteGuard; None no guard
            closed: Per axis, whether the walls are closed
            wall_keep: The share along a closed wall kept this step (slip_time's); 1 free-slip
            clamp: CORRECTION: where it overshoots, cut it to the bounds instead of reverting to the predictor
        """
        if not self.allocated or not self.shader_program:
            return
        glUseProgram(self.shader_program)
        bind_faces(self, source, cells, cell, closed, has_obstacles, obstacle)
        bind_face_writes(self, result)
        if predictor is not None:
            self.bind_texture_3d(8, predictor, 'uPred')
        glUniform1i(self.get_uniform_loc('uMode'), mode)
        glUniform1i(self.get_uniform_loc('uClamp'), int(clamp))
        glUniform1f(self.get_uniform_loc('uTimestep'), step)
        glUniform1f(self.get_uniform_loc('uDissipation'), dissipation)
        glUniform1f(self.get_uniform_loc('uEvaporation'), evaporation)
        glUniform1f(self.get_uniform_loc('uWallKeep'), wall_keep)
        if guard is not None:
            guard.use(self.get_uniform_loc('uGuardSlot'))
        else:
            glUniform1i(self.get_uniform_loc('uGuardSlot'), -1)
        self.dispatch(cells[0] + 1, cells[1] + 1, cells[2] + 1)
