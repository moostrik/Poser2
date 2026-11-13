"""The warp from the sensor's frame to the delivered one: its mesh, and the rows it covers per column."""

import math

import numpy as np

from .frame_window import FrameWindow, frame_window
from .lens import degrees_per_pixel, source_lens

WARP_MESH: int = 32             # mesh points per side: 16 is already sub-pixel, 32 holds 0.23 px on a 4056 px frame


def _project_grid(
    src_size: tuple[int, int],
    out_size: tuple[int, int],
    mode_width: int,
    fov_h: float,
    tilt: float,
    flip_h: bool,
    flip_v: bool,
    xs: np.ndarray,
    ys: np.ndarray,
    lens_fov: float,
    lens_center: tuple[float, float],
) -> np.ndarray:
    """The source pixel each output pixel of the grid `ys x xs` reads: (len(ys), len(xs), 2)."""
    src_w, src_h = src_size
    out_w, out_h = out_size
    lens_focal, cx, cy = source_lens(src_size, mode_width, fov_h, lens_fov, lens_center)
    window: FrameWindow = frame_window(src_size, out_size, mode_width, fov_h, tilt, lens_fov, lens_center)
    X, Y = np.meshgrid(np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64))
    if window.focal <= 0.0 or lens_focal <= 0.0:
        # No field to speak of: hand the frame through, the square crop cut from the centre.
        return np.stack([X + (src_w - out_w) / 2.0, Y + (src_h - out_h) / 2.0], axis=-1)

    ox: np.ndarray = (out_w - 1.0) - X if flip_h else X
    oy: np.ndarray = (out_h - 1.0) - Y if flip_v else Y
    # Output pixel -> world ray. Image y is down, so d1 is the DOWNWARD component.
    bearing: np.ndarray = (ox - (out_w - 1.0) / 2.0) * math.radians(degrees_per_pixel(fov_h, mode_width))
    elev: np.ndarray = np.arctan((window.horizon_px - oy) / window.focal)
    cos_e: np.ndarray = np.cos(elev)
    d0: np.ndarray = cos_e * np.sin(bearing)
    d1: np.ndarray = -np.sin(elev)
    d2: np.ndarray = cos_e * np.cos(bearing)
    # Undo the tilt: rotate about the horizontal axis by -tilt (the camera is aimed UP by tilt).
    t: float = math.radians(-tilt)
    cos_t, sin_t = math.cos(t), math.sin(t)
    d1, d2 = cos_t * d1 - sin_t * d2, sin_t * d1 + cos_t * d2
    # Ray -> source pixel, equidistant: radius is the polar angle off the axis times the focal.
    r: np.ndarray = np.arccos(np.clip(d2, -1.0, 1.0)) * lens_focal
    psi: np.ndarray = np.arctan2(d1, d0)
    return np.stack([cx + r * np.cos(psi), cy + r * np.sin(psi)], axis=-1)


def warp_mesh_points(
    src_size: tuple[int, int],
    out_size: tuple[int, int],
    mode_width: int,
    fov_h: float,
    tilt: float,
    flip_h: bool = False,
    flip_v: bool = False,
    mesh_w: int = WARP_MESH,
    mesh_h: int = WARP_MESH,
    lens_fov: float = 0.0,
    lens_center: tuple[float, float] = (0.0, 0.0),
) -> list[tuple[float, float]]:
    """The warp's mesh, row-major: the source pixel each output grid point reads, a column one bearing and a
    row one tangent of elevation, the up-tilt (°) undone."""
    out_w, out_h = out_size
    grid: np.ndarray = _project_grid(
        src_size, out_size, mode_width, fov_h, tilt, flip_h, flip_v,
        np.linspace(0.0, out_w - 1.0, mesh_w), np.linspace(0.0, out_h - 1.0, mesh_h),
        lens_fov, lens_center)
    return [(float(x), float(y)) for x, y in grid.reshape(-1, 2)]


def frame_coverage(
    src_size: tuple[int, int],
    out_size: tuple[int, int],
    mode_width: int,
    fov_h: float,
    tilt: float,
    flip_h: bool = False,
    flip_v: bool = False,
    lens_fov: float = 0.0,
    lens_center: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """Per output column the first and last row the sensor covers, in delivered order: (out_w, 2) int,
    (-1, -1) for a column without picture. Tens of ms: for open time."""
    src_w, src_h = src_size
    out_w, out_h = out_size
    pts: np.ndarray = _project_grid(
        src_size, out_size, mode_width, fov_h, tilt, flip_h, flip_v,
        np.arange(out_w), np.arange(out_h), lens_fov, lens_center)
    # A pixel's footprint is half a pixel either side of its index; the edge rows and columns
    # count as covered up to that, which also keeps float noise on an exact edge from losing them.
    inside: np.ndarray = np.asarray((pts[..., 0] >= -0.5) & (pts[..., 0] <= src_w - 0.5)
                                    & (pts[..., 1] >= -0.5) & (pts[..., 1] <= src_h - 0.5))
    # The black is an arch at the top or the bottom, so a column's covered rows are one run.
    covered: np.ndarray = np.asarray(inside.any(axis=0))
    first: np.ndarray = inside.argmax(axis=0)
    last: np.ndarray = out_h - 1 - inside[::-1, :].argmax(axis=0)
    return np.stack([np.where(covered, first, -1), np.where(covered, last, -1)], axis=1).astype(int)


def coverage_summary(coverage: np.ndarray, out_size: tuple[int, int], mode_width: int,
                     fov_h: float, flip_h: bool = False) -> str:
    """One line for the open log: the rows covered at the center column, the seams (45° off axis, the worse
    side) and the edge columns."""
    out_w, out_h = out_size
    dpp: float = degrees_per_pixel(fov_h, mode_width)
    center: float = (out_w - 1) / 2.0

    def column(bearing: float) -> int:
        x: float = center +(bearing / dpp if dpp > 0.0 else 0.0)
        x = (out_w - 1) - x if flip_h else x
        return int(round(min(max(x, 0.0), out_w - 1.0)))

    def span(columns: list[int]) -> str:
        rows = coverage[columns]
        if (rows[:, 0] < 0).any():
            return 'none'
        return f'{int(rows[:, 0].max())}-{int(rows[:, 1].min())}'

    return (f'rows covered of {out_h}: centre {span([column(0.0)])}, '
            f'seams {span([column(-45.0), column(45.0)])}, '
            f'edges {span([0, out_w - 1])}')
