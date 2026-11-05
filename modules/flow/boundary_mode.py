"""The boundary condition of a fluid field at the edge of its grid, shared by the 2D and the 3D fluid."""

from enum import IntEnum


class BoundaryMode(IntEnum):
    OPEN   = 0  # Dirichlet — zero value at boundary (field escapes / absorbing)
    CLOSED = 1  # Neumann   — zero gradient at boundary (field reflects / insulating)
