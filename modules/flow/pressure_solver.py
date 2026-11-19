"""The pressure solver: how a fluid's Poisson equation is solved, shared by the 2D and the 3D fluid."""

from enum import IntEnum


class PressureSolver(IntEnum):
    """How the pressure's Poisson equation is solved each step, warm from the last step's pressure."""
    JACOBI = 0              # every cell from its neighbors' last values; five iterations a dispatch in shared memory
    SOR = 1                 # red-black Gauss–Seidel, over-relaxed: each color from the other's newest values
    MULTIGRID = 2           # V-cycles over the grid halved to a few cells, red-black Gauss–Seidel smoothing each level
