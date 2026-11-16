"""The depth in the view as numpy, what `DepthCrop`, `DepthUpscale` and `DepthFill` do on the GPU, for the tests and
the harness: the camera's depth inside the view's box on its own pixels (`depth_crop`), the depth upscaled to the
view, finer than its own pixels, where it is the player's and the mask is (`upscaled_depth`, on
`edge_aware_depth`), and that filled across the mask by push-pull (`push_pull`). A depth is (H, W) metres, 0 for none, rows down;
uv is rows down."""

import numpy as np


def depth_crop(depth: np.ndarray, u: np.ndarray, v: np.ndarray, near: float, far: float) -> np.ndarray:
    """The depth inside the box at uv on the depth's own pixels: the depth pixel the point lies in where it is from
    `near` to `far` (m), else 0; no depth (0) and beyond the grid are 0. Mirrors DepthCrop."""
    height, width = depth.shape
    ix: np.ndarray = np.floor(u * width).astype(np.int64)
    iy: np.ndarray = np.floor(v * height).astype(np.int64)
    inside: np.ndarray = (ix >= 0) & (iy >= 0) & (ix < width) & (iy < height)
    d: np.ndarray = np.where(inside, depth[np.clip(iy, 0, height - 1), np.clip(ix, 0, width - 1)], 0.0)
    return np.where((d > 0.0) & (d >= near) & (d <= far), d, 0.0)


def edge_aware_depth(depth: np.ndarray, u: np.ndarray, v: np.ndarray, tolerance: float) -> np.ndarray:
    """The depth read at uv: the four depth pixels around a point, interpolated bilinearly where they lie within
    `tolerance` (m) of each other, else the nearest, the pixel the point lies in; a 0, no depth, is never blended,
    so with one of the four 0 the nearest is taken. Beyond the grid is no depth."""
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


def upscaled_depth(depth: np.ndarray, u: np.ndarray, v: np.ndarray, tolerance: float, mask: np.ndarray,
                   near: float, far: float) -> np.ndarray:
    """`edge_aware_depth` where it is the player's, a depth from `near` to `far` (m), and `mask` (0 to 1, at the
    point) is above 0; else 0. Mirrors DepthUpscale."""
    surface: np.ndarray = edge_aware_depth(depth, u, v, tolerance)
    return np.where((surface > 0.0) & (surface >= near) & (surface <= far) & (mask > 0.0), surface, 0.0)


def _push(pair: np.ndarray) -> np.ndarray:
    """One level down: each pixel the mean of the up to four it covers, the pairs (depth × weight, weight) averaged
    as they are; the size halved, rounded up. Pairs are grouped from the bottom row, as GL holds a texture."""
    height, width = pair.shape[:2]
    h, w = (height + 1) // 2, (width + 1) // 2
    flipped: np.ndarray = pair[::-1]
    padded: np.ndarray = np.zeros((h * 2, w * 2, 2), pair.dtype)
    padded[:height, :width] = flipped
    count: np.ndarray = np.zeros((h * 2, w * 2), pair.dtype)
    count[:height, :width] = 1.0
    sums: np.ndarray = padded.reshape(h, 2, w, 2, 2).sum(axis=(1, 3))
    counts: np.ndarray = count.reshape(h, 2, w, 2).sum(axis=(1, 3))
    return (sums / counts[..., None])[::-1]


def _coarse(pair: np.ndarray, height: int, width: int) -> np.ndarray:
    """The coarser level's pairs at each pixel of the finer (`height` × `width`), bilinearly, its pixel centers at
    the middle of the two it covers, clamped at its edges. Rows counted from the bottom, as GL holds a texture."""
    ch, cw = pair.shape[:2]
    flipped: np.ndarray = pair[::-1]
    y: np.ndarray = (np.arange(height) + 0.5) / 2.0 - 0.5
    x: np.ndarray = (np.arange(width) + 0.5) / 2.0 - 0.5
    y0: np.ndarray = np.floor(y).astype(np.int64)
    x0: np.ndarray = np.floor(x).astype(np.int64)
    fy: np.ndarray = (y - y0)[:, None, None]
    fx: np.ndarray = (x - x0)[None, :, None]
    ya, yb = np.clip(y0, 0, ch - 1), np.clip(y0 + 1, 0, ch - 1)
    xa, xb = np.clip(x0, 0, cw - 1), np.clip(x0 + 1, 0, cw - 1)
    top: np.ndarray = flipped[ya][:, xa] * (1 - fx) + flipped[ya][:, xb] * fx
    bottom: np.ndarray = flipped[yb][:, xa] * (1 - fx) + flipped[yb][:, xb] * fx
    return (top * (1 - fy) + bottom * fy)[::-1]


def pyramid_sizes(height: int, width: int) -> list[tuple[int, int]]:
    """The push's levels below the first, (height, width), each half the one above rounded up, down to 1 × 1."""
    sizes: list[tuple[int, int]] = []
    while height > 1 or width > 1:
        height, width = (height + 1) // 2, (width + 1) // 2
        sizes.append((height, width))
    return sizes


def push_pull(read: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """`read` (metres, 0 for none) filled where `mask` (0 to 1) is above 0 (Gortler et al. 1996): a pixel with a
    depth keeps it; one without takes the depth around it, from the coarsest level of the pyramid that has one.
    Push: each level the mean of the pairs (depth × weight, weight) below, the first's weight 1 where there is a
    depth. Pull: from the coarsest up, a level's pair plus (1 − its weight) times the coarser level's, read
    bilinearly. Where the mask is 0, 0. Mirrors DepthFill."""
    present: np.ndarray = (read > 0.0).astype(np.float64)
    levels: list[np.ndarray] = [np.stack([read * present, present], axis=-1)]
    for _ in pyramid_sizes(*read.shape):
        levels.append(_push(levels[-1]))
    pulled: np.ndarray = levels[-1]
    for level in reversed(levels[:-1]):
        coarse: np.ndarray = _coarse(pulled, *level.shape[:2])
        pulled = level + (1.0 - level[..., 1:2]) * coarse
    weight: np.ndarray = pulled[..., 1]
    filled: np.ndarray = np.where(weight > 0.0, pulled[..., 0] / np.maximum(weight, 1e-30), 0.0)
    return np.where(mask > 0.0, filled, 0.0)
