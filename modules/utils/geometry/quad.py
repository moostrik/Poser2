from dataclasses import dataclass

from .point import Point2f


@dataclass
class Quad:
    """Four corners in 0..1 with the origin top-left: where a rectangle's top left, top right, bottom right and bottom
    left land, as a turned or mirrored rectangle."""
    top_left: Point2f
    top_right: Point2f
    bottom_right: Point2f
    bottom_left: Point2f
