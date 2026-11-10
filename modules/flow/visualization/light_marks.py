"""Where a volume's light is, drawn in its box as boxes `CubeEdges` outlines: a lamp as a small cube at its
place; a sun as a line from the box's center out along the way its light comes, past where it enters,
the small cube at the line's outer end. Plain numpy, so it tests without GL.

Each mark is a matrix that takes the unit box (-0.5 to 0.5 each way) to the mark, in the box's space
(`orbit`: x right, y up, the front layer at +z)."""

import numpy as np

from ..port import LightPlace

MARK_SIZE: float = 0.04         # the small cube's side, in box widths
LINE_WIDTH: float = 0.004       # the sun's line, as a thin box, in box widths
SUN_REACH: float = 1.6          # the sun's line ends this far out, as a share of the way to where it enters


def box_point(uvw: tuple[float, float, float], box: tuple[float, float, float]) -> np.ndarray:
    """Texture coordinates (z 0 the front) as a point in the box's space, as the cube reads its volume."""
    u, v, w = uvw
    return np.array([(u - 0.5) * box[0], (v - 0.5) * box[1], (0.5 - w) * box[2]])


def _cube_at(center: np.ndarray, size: float) -> np.ndarray:
    mark: np.ndarray = np.eye(4)
    mark[:3, :3] *= size
    mark[:3, 3] = center
    return mark


def _line(start: np.ndarray, end: np.ndarray, width: float) -> np.ndarray:
    """A box `width` thick from `start` to `end`."""
    along: np.ndarray = end - start
    length: float = float(np.linalg.norm(along))
    if length < 1e-9:
        return _cube_at(start, width)
    axis: np.ndarray = along / length
    helper: np.ndarray = np.array([0.0, 1.0, 0.0]) if abs(axis[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    side: np.ndarray = np.cross(axis, helper)
    side /= np.linalg.norm(side)
    up: np.ndarray = np.cross(axis, side)
    mark: np.ndarray = np.eye(4)
    mark[:3, 0] = side * width
    mark[:3, 1] = up * width
    mark[:3, 2] = along
    mark[:3, 3] = 0.5 * (start + end)
    return mark


def light_marks(light: LightPlace, box: tuple[float, float, float]) -> list[np.ndarray]:
    """The marks that show `light` in a box of `box`'s size."""
    place: np.ndarray = box_point(light.uvw, box)
    if light.point:
        return [_cube_at(place, MARK_SIZE)]
    outer: np.ndarray = place * SUN_REACH
    return [_line(np.zeros(3), outer, LINE_WIDTH), _cube_at(outer, MARK_SIZE)]
