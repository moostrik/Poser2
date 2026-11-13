"""The depth camera's pixels on the video frame: the projection the GPU aligns depth with, and its size."""

import numpy as np

from ..frame import DepthAlignment


def depth_projection(alignment: DepthAlignment | None, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """For a depth frame of `size`: `to_ray` (3x3), a pixel (x, y, 1) to its ray at unit depth, and `to_uv`
    (3x4), a point on it to the video frame's homogeneous uv, its third coordinate the depth along the video
    camera's axis; pixel i's center at (i + 0.5) / width."""
    width, height = size
    if alignment is None:
        to_ray: np.ndarray = np.array([[1.0 / width, 0.0, 0.5 / width], [0.0, 1.0 / height, 0.5 / height], [0.0, 0.0, 1.0]])
        return to_ray, np.hstack([np.eye(3), np.zeros((3, 1))])
    color_w, color_h = alignment.color_size
    lens: np.ndarray = np.array(alignment.color_intrinsics, dtype=np.float64) + np.array([[0.0, 0.0, 0.5], [0.0, 0.0, 0.5], [0.0, 0.0, 0.0]])
    to_uv: np.ndarray = np.diag([1.0 / color_w, 1.0 / color_h, 1.0]) @ lens @ np.array(alignment.to_color, dtype=np.float64)[:3]
    return np.linalg.inv(np.array(alignment.intrinsics, dtype=np.float64)), to_uv


def aligned_depth_size(size: tuple[int, int], alignment: DepthAlignment | None) -> tuple[int, int]:
    """The size of depth on the video's view: its own rows, and an even width at the video's aspect."""
    if alignment is None:
        return size
    color_w, color_h = alignment.color_size
    return (round(size[1] * color_w / color_h / 2.0) * 2, size[1])
