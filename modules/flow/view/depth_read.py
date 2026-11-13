"""The depth read into the view, finer than its own pixels: `DepthRead`'s ways as numpy, what `DepthUpscale` does on
the GPU, for the tests and the harness. A depth is (H, W) metres, 0 for none, rows down; uv is rows down."""

import math
from enum import IntEnum

import numpy as np


class DepthRead(IntEnum):
    """How the depth is read into the view, finer than its own pixels."""
    SURFACE = 0         # interpolated within a surface, the nearest across an edge: the edge in the depth's pixels
    MASK = 1            # as SURFACE, an edge's side taken from the mask: the edge on the mask's, the image's
    BILATERAL = 2       # joint bilateral upsampling with the camera image (Kopf et al. 2007)


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


def mask_sided_depth(depth: np.ndarray, u: np.ndarray, v: np.ndarray, tolerance: float, inside: np.ndarray,
                     near: float, far: float, reach: float) -> np.ndarray:
    """`edge_aware_depth` where it lies on the side `inside` (bool, the mask at the point) gives: the player's, a
    depth from `near` to `far` (m), where it is True, anything else (no depth included) where it is False; else the
    nearest depth pixel on that side within `reach` depth pixels of the point, else `edge_aware_depth`. Mirrors
    DepthUpscale's MASK."""
    height, width = depth.shape

    def player(d: np.ndarray) -> np.ndarray:
        return (d > 0.0) & (d >= near) & (d <= far)

    surface: np.ndarray = edge_aware_depth(depth, u, v, tolerance)
    wrong: np.ndarray = player(surface) != inside
    px: np.ndarray = u * width - 0.5
    py: np.ndarray = v * height - 0.5
    cx: np.ndarray = np.floor(px + 0.5).astype(np.int64)
    cy: np.ndarray = np.floor(py + 0.5).astype(np.int64)
    r: int = math.ceil(reach + 0.5)
    best: np.ndarray = np.full(np.shape(u), reach * reach)
    picked: np.ndarray = surface.copy()
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            ax, ay = cx + dx, cy + dy
            in_grid: np.ndarray = (ax >= 0) & (ay >= 0) & (ax < width) & (ay < height)
            d: np.ndarray = np.where(in_grid, depth[np.clip(ay, 0, height - 1), np.clip(ax, 0, width - 1)], 0.0)
            distance: np.ndarray = (ax - px) ** 2 + (ay - py) ** 2
            take: np.ndarray = wrong & (player(d) == inside) & (distance <= best)
            picked = np.where(take, d, picked)
            best = np.where(take, distance, best)
    return picked


def joint_bilateral_depth(depth: np.ndarray, image: np.ndarray, u: np.ndarray, v: np.ndarray, contrast: float,
                          radius: int = 2) -> np.ndarray:
    """Joint bilateral upsampling (Kopf et al. 2007) of the depth at uv, guided by `image` (h, w, 3, 0 to 1, rows
    down, on the depth's view, read bilinearly at uv): the depth pixels within `radius` of the point, each weighed
    by a gaussian of its distance (half a depth pixel, the paper's σd) and of the image's color distance at it and
    at the point (`contrast`, σr); a 0 is left out. Mirrors DepthUpscale's BILATERAL."""
    height, width = depth.shape

    def color(uu: np.ndarray, vv: np.ndarray) -> np.ndarray:
        ih, iw = image.shape[:2]
        x = np.clip(uu * iw - 0.5, 0, iw - 1)
        y = np.clip(vv * ih - 0.5, 0, ih - 1)
        x0 = np.floor(x).astype(np.int64)
        y0 = np.floor(y).astype(np.int64)
        x1 = np.minimum(x0 + 1, iw - 1)
        y1 = np.minimum(y0 + 1, ih - 1)
        fx = (x - x0)[..., None]
        fy = (y - y0)[..., None]
        top = image[y0, x0] * (1 - fx) + image[y0, x1] * fx
        bottom = image[y1, x0] * (1 - fx) + image[y1, x1] * fx
        return top * (1 - fy) + bottom * fy

    px: np.ndarray = u * width - 0.5
    py: np.ndarray = v * height - 0.5
    cx: np.ndarray = np.floor(px + 0.5).astype(np.int64)
    cy: np.ndarray = np.floor(py + 0.5).astype(np.int64)
    here: np.ndarray = color(u, v)
    total: np.ndarray = np.zeros(np.shape(u))
    weights: np.ndarray = np.zeros(np.shape(u))
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            ax, ay = cx + dx, cy + dy
            inside: np.ndarray = (ax >= 0) & (ay >= 0) & (ax < width) & (ay < height)
            d: np.ndarray = np.where(inside, depth[np.clip(ay, 0, height - 1), np.clip(ax, 0, width - 1)], 0.0)
            there: np.ndarray = color((ax + 0.5) / width, (ay + 0.5) / height) - here
            w: np.ndarray = np.exp(-((ax - px) ** 2 + (ay - py) ** 2) * 2.0 - np.sum(there ** 2, axis=-1) * 0.5 / max(contrast ** 2, 1e-8))
            w = np.where(d > 0.0, w, 0.0)
            total += w * d
            weights += w
    return np.where(weights > 1e-12, total / np.maximum(weights, 1e-12), 0.0)
