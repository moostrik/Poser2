"""The wall: a fluid field's condition at the edge of its grid, shared by the 2D and the 3D fluid."""

from enum import IntEnum


class WallMode(IntEnum):
    OPEN   = 0  # Dirichlet — zero value at the wall (the field escapes)
    CLOSED = 1  # Neumann   — zero gradient at the wall (the field reflects)
