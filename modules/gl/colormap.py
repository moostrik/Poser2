"""Colormap: the colormaps `shaders/colormap.glsl` holds, by the number a shader takes."""

from enum import IntEnum


class Colormap(IntEnum):
    """A value 0..1 to a color: turbo and viridis, the textbook ones now; jet, the old rainbow."""
    TURBO   = 0
    VIRIDIS = 1
    JET     = 2
