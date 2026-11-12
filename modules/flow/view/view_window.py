"""The view's window on the camera image and the depth read into it, as plain geometry and arrays.

The window is uv on the camera image with rows down (`Rect`, x and y its top left corner), as the image and the
aligned depth are both on the image's view. `texture_roi` gives it with rows up, as `DrawRoi` and the shaders
read a texture. `edge_aware_depth` is what `DepthUpscale` does on the GPU, for the tests and the harness.
"""

import math

import numpy as np

from modules.utils import Rect


def fixed_window(center_x: float, center_y: float, height: float, output_aspect: float, cam_aspect: float) -> Rect:
    """A window of the camera image (uv, y down) centered at (`center_x`, `center_y`), `height` of the
    image's height tall, `output_aspect` wide over tall in pixels; `cam_aspect` is the image's width over its height."""
    width: float = height * output_aspect / cam_aspect
    return Rect(center_x - width / 2.0, center_y - height / 2.0, width, height)


def texture_roi(window: Rect) -> Rect:
    """The window with rows up, as a texture is read (`DrawRoi`'s roi)."""
    return Rect(window.x, 1.0 - window.y - window.height, window.width, window.height)


def view_span(window: Rect, fov: float) -> float:
    """The metres one width of the view spans per metre of depth: the window's share of the camera image's
    horizontal field of view `fov` (degrees); 0 without a field of view."""
    return window.width * 2.0 * math.tan(math.radians(fov) / 2.0) if fov > 0.0 else 0.0


def depth_step(window: Rect, grid_size: tuple[int, int]) -> tuple[float, float]:
    """The uv between two of the depth's own samples in the view, for a depth on the camera image at `grid_size`
    (width, height); (0, 0) while the window or the grid has no size."""
    width, height = grid_size
    if window.width <= 0.0 or window.height <= 0.0 or width <= 0 or height <= 0:
        return (0.0, 0.0)
    return (1.0 / width / window.width, 1.0 / height / window.height)


def view_per_depth_pixel(window: Rect, grid_height: int, view_height: int) -> float:
    """The view's rows one row of the depth's grid spans; 1 while the window or the grid has no size."""
    depth_rows: float = window.height * grid_height
    return view_height / depth_rows if depth_rows > 0.0 else 1.0


def place_points(x: np.ndarray, y: np.ndarray, bbox: Rect, window: Rect) -> tuple[np.ndarray, np.ndarray]:
    """Points relative to `bbox` (0 to 1 across it, rows down) as uv in the window, rows down."""
    if window.width <= 0.0 or window.height <= 0.0:
        return np.full_like(x, np.nan), np.full_like(y, np.nan)
    image_x: np.ndarray = x * bbox.width + bbox.x
    image_y: np.ndarray = y * bbox.height + bbox.y
    return (image_x - window.x) / window.width, (image_y - window.y) / window.height


def edge_aware_depth(depth: np.ndarray, u: np.ndarray, v: np.ndarray, tolerance: float) -> np.ndarray:
    """The depth (H, W metres, 0 for none, rows down) read at uv (rows down): the four depth pixels around a point,
    interpolated bilinearly where they lie within `tolerance` (m) of each other, else the nearest, the pixel the
    point lies in; a 0, no depth, is never blended, so with one of the four 0 the nearest is taken. Beyond the
    grid is no depth."""
    height, width = depth.shape

    def fetch(ix: np.ndarray, iy: np.ndarray) -> np.ndarray:
        inside: np.ndarray = (ix >= 0) & (iy >= 0) & (ix < width) & (iy < height)
        return np.where(inside, depth[np.clip(iy, 0, height - 1), np.clip(ix, 0, width - 1)], 0.0)

    px: np.ndarray = u * width - 0.5
    py: np.ndarray = v * height - 0.5
    x0: np.ndarray = np.floor(px).astype(np.int64)
    y0: np.ndarray = np.floor(py).astype(np.int64)
    fx: np.ndarray = px - x0
    fy: np.ndarray = py - y0
    d00, d10, d01, d11 = fetch(x0, y0), fetch(x0 + 1, y0), fetch(x0, y0 + 1), fetch(x0 + 1, y0 + 1)
    low: np.ndarray = np.minimum(np.minimum(d00, d10), np.minimum(d01, d11))
    high: np.ndarray = np.maximum(np.maximum(d00, d10), np.maximum(d01, d11))
    blended: np.ndarray = (d00 * (1 - fx) + d10 * fx) * (1 - fy) + (d01 * (1 - fx) + d11 * fx) * fy
    nearest: np.ndarray = fetch(np.floor(u * width).astype(np.int64), np.floor(v * height).astype(np.int64))
    return np.where((low > 0.0) & (high - low <= tolerance), blended, nearest)
