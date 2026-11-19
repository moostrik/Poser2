"""The advection: how a fluid carries a field each step, and MacCormack's limiter; shared by the 2D and the 3D fluid."""

from enum import IntEnum


class Advection(IntEnum):
    """How the velocity, the density and the temperature are carried each step."""
    SEMI_LAGRANGIAN = 0     # trace back and read, trilinear: stable, blurs a little every step
    MACCORMACK = 1          # the same, corrected by half its error traced forward, clamped (Selle 2008)


class Limiter(IntEnum):
    """What MacCormack does where its correction leaves the eight texels the plain step read between (Selle 2008)."""
    CLAMP = 0       # the corrected value cut to those bounds: the edge kept steep, its top flat
    REVERT = 1      # the plain step's value there: what a step really made, softer at the sharpest features
