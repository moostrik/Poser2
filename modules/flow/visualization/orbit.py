"""The orbit camera of a monitor's 3D views: yaw, pitch and zoom around the volume's box, and the
matrices that draw it. Plain numpy, so it tests without GL.

The box is the volume's shape with its width 1: x from -0.5 to 0.5, y half its height ratio up and
down, z half its depth (`extent`) toward and away from the viewer. The front face, the volume's layer
0, is at +z, toward the camera of the front view; the back face, its last layer, at -z."""

import math

import numpy as np

FOV: float = math.radians(30.0)     # the vertical field of view
PITCH_LIMIT: float = 89.0           # degrees: never over the top, where up would flip
ZOOM_LIMITS: tuple[float, float] = (0.4, 4.0)
TURN: float = 180.0                 # degrees a drag across the whole window turns
ZOOM_RATE: float = 2.0              # a drag over the window's height zooms by e^2


def box_size(width: int, height: int, extent: float) -> tuple[float, float, float]:
    """The box's size with its width 1: (1, height / width, extent)."""
    return (1.0, height / max(width, 1), max(extent, 1e-6))


class Orbit:
    """Yaw and pitch (degrees) and zoom (1 frames the box) of a camera looking at the box's center."""

    def __init__(self, yaw: float = 0.0, pitch: float = 0.0, zoom: float = 1.0) -> None:
        self.yaw: float = yaw
        self.pitch: float = pitch
        self.zoom: float = zoom

    def reset(self) -> None:
        """The front view, framed."""
        self.yaw, self.pitch, self.zoom = 0.0, 0.0, 1.0

    def turn(self, dx: float, dy: float) -> None:
        """A drag of `dx`, `dy` (shares of the window, y down): the box turns with the drag."""
        self.yaw = (self.yaw - dx * TURN + 180.0) % 360.0 - 180.0
        self.pitch = min(max(self.pitch + dy * TURN, -PITCH_LIMIT), PITCH_LIMIT)

    def dolly(self, dy: float) -> None:
        """A drag of `dy` (a share of the window's height, down positive): up comes closer."""
        self.zoom = min(max(self.zoom * math.exp(dy * ZOOM_RATE), ZOOM_LIMITS[0]), ZOOM_LIMITS[1])

    def eye(self, box: tuple[float, float, float], aspect: float = 1.0) -> np.ndarray:
        """Where the camera is: on a sphere around the center, far enough that the box fits a view
        `aspect` wide from any angle at zoom 1, by the narrower of its two fields of view."""
        radius: float = 0.5 * math.sqrt(sum(s * s for s in box))
        across: float = 2.0 * math.atan(math.tan(FOV / 2.0) * max(aspect, 1e-6))
        distance: float = self.zoom * radius / math.sin(min(FOV, across) / 2.0)
        yaw, pitch = math.radians(self.yaw), math.radians(self.pitch)
        return np.array([math.cos(pitch) * math.sin(yaw), math.sin(pitch), math.cos(pitch) * math.cos(yaw)]) * distance

    def front_on_near_side(self) -> bool:
        """Whether the camera is on the front face's side of the box, so the back layers are the far ones."""
        return math.cos(math.radians(self.pitch)) * math.cos(math.radians(self.yaw)) >= 0.0

    def view_projection(self, box: tuple[float, float, float], aspect: float) -> np.ndarray:
        """The matrix from the box's space to clip space (4 × 4, row-major) for a view `aspect` wide."""
        eye: np.ndarray = self.eye(box, aspect)
        return perspective(FOV, aspect, 0.05, float(np.linalg.norm(eye)) * 4.0) @ look_at(eye, np.zeros(3), np.array([0.0, 1.0, 0.0]))


AXIS_CORNER: tuple[float, float, float] = (-0.5, -0.5, 0.5)     # x 0, y 0, the front layer, in shares of the box
AXIS_ENDS: tuple[tuple[str, tuple[float, float, float]], ...] = (
    ('x', (0.5, -0.5, 0.5)), ('y', (-0.5, 0.5, 0.5)), ('z', (-0.5, -0.5, -0.5)))
LABEL_REACH: float = 1.1    # a letter sits past its axis's end, by a share of the axis


def axis_labels(view_projection: np.ndarray, box: tuple[float, float, float], width: int, height: int) -> list[tuple[str, float, float]]:
    """Each axis's letter and where it goes in a view `width` × `height`: in pixels from the top left,
    just past the end of the box's edge along that axis from its first corner; none behind the camera."""
    size: np.ndarray = np.asarray(box)
    corner: np.ndarray = np.asarray(AXIS_CORNER) * size
    labels: list[tuple[str, float, float]] = []
    for name, end in AXIS_ENDS:
        tip: np.ndarray = corner + (np.asarray(end) * size - corner) * LABEL_REACH
        clip: np.ndarray = view_projection @ np.append(tip, 1.0)
        if clip[3] <= 1e-6:
            continue
        x, y = clip[0] / clip[3], clip[1] / clip[3]
        labels.append((name, (x + 1.0) * 0.5 * width, (1.0 - y) * 0.5 * height))
    return labels


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """The view matrix of a camera at `eye` looking at `target` (row-major)."""
    forward: np.ndarray = target - eye
    forward /= np.linalg.norm(forward)
    side: np.ndarray = np.cross(forward, up)
    if np.linalg.norm(side) < 1e-6:                 # looking straight up or down: any side will do
        side = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    side /= np.linalg.norm(side)
    true_up: np.ndarray = np.cross(side, forward)
    view: np.ndarray = np.identity(4)
    view[0, :3], view[1, :3], view[2, :3] = side, true_up, -forward
    view[:3, 3] = -view[:3, :3] @ eye
    return view


def perspective(fov: float, aspect: float, near: float, far: float) -> np.ndarray:
    """The projection matrix (row-major), as gluPerspective."""
    f: float = 1.0 / math.tan(fov / 2.0)
    projection: np.ndarray = np.zeros((4, 4))
    projection[0, 0] = f / max(aspect, 1e-6)
    projection[1, 1] = f
    projection[2, 2] = (far + near) / (near - far)
    projection[2, 3] = 2.0 * far * near / (near - far)
    projection[3, 2] = -1.0
    return projection
