"""The density's blend, both fluids': how the body's dye goes into the water (`shaders/density_blend.glsl`)."""

from enum import IntEnum


class DensityBlend(IntEnum):
    """How the body's dye goes into the water, premultiplied: over it, compositing (Porter and Duff 1984),
    added to it, a source (Stam 2003), or the water drawn toward it, kept to compare."""
    OVER = 0        # the dye over the water's by the drag's share: the amount never falls and never passes 1
    ADD = 1         # the dye × the gain × dt added; at more than fully dyed all four channels scaled down to it
    FILL = 2        # the water drawn toward the dye, at most fully dyed, its color too: up by the drag's share, down by the release's
