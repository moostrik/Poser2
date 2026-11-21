"""Boxes, keypoints, lenses and depth alignments as they are in the frame flipped left to right; boxes and keypoints
also in the frame flipped top to bottom and turned (transposed)."""

import numpy as np

from modules.utils import Rect

from ..frame import DepthAlignment

MIRROR_X: np.ndarray = np.diag([-1.0, 1.0, 1.0, 1.0])     # a 4x4 move's x turned over


def mirrored_roi(roi: Rect) -> Rect:
    """A normalised box in the frame flipped left to right."""
    return Rect(1.0 - roi.x - roi.width, roi.y, roi.width, roi.height)


def mirrored_keypoints(keypoints: np.ndarray | None) -> np.ndarray | None:
    """Keypoints (N, 3) with x flipped; the labels stay the model's."""
    if keypoints is None:
        return None
    mirrored: np.ndarray = keypoints.copy()
    mirrored[:, 0] = 1.0 - mirrored[:, 0]
    return mirrored


def flipped_roi(roi: Rect) -> Rect:
    """A normalised box in the frame flipped top to bottom."""
    return Rect(roi.x, 1.0 - roi.y - roi.height, roi.width, roi.height)


def flipped_keypoints(keypoints: np.ndarray | None) -> np.ndarray | None:
    """Keypoints (N, 3) with y flipped; the labels stay the model's."""
    if keypoints is None:
        return None
    flipped: np.ndarray = keypoints.copy()
    flipped[:, 1] = 1.0 - flipped[:, 1]
    return flipped


def turned_roi(roi: Rect) -> Rect:
    """A normalised box in the frame turned as `CameraFrame.rotate90` turns it: transposed."""
    return Rect(roi.y, roi.x, roi.height, roi.width)


def turned_keypoints(keypoints: np.ndarray | None) -> np.ndarray | None:
    """Keypoints (N, 3) in the transposed frame: x and y swapped; the labels stay the model's."""
    if keypoints is None:
        return None
    turned: np.ndarray = keypoints.copy()
    turned[:, [0, 1]] = keypoints[:, [1, 0]]
    return turned


def mirrored_intrinsics(intrinsics: list[list[float]], width: int) -> list[list[float]]:
    """A 3x3 lens as it is for the frame `width` px flipped left to right."""
    mirrored: list[list[float]] = [list(row) for row in intrinsics]
    mirrored[0][2] = width - mirrored[0][2]
    return mirrored


def mirrored_alignment(alignment: DepthAlignment, width: int) -> DepthAlignment:
    """The alignment of a depth frame `width` px flipped left to right: both lenses' centers mirrored and the
    move turned with them."""
    def mirrored(lens: list[list[float]], size: int) -> list[list[float]]:
        flipped: list[list[float]] = [list(row) for row in lens]
        flipped[0][2] = size - 1 - flipped[0][2]
        return flipped
    to_color: np.ndarray = MIRROR_X @ np.array(alignment.to_color, dtype=np.float64) @ MIRROR_X
    return DepthAlignment(mirrored(alignment.intrinsics, width), mirrored(alignment.color_intrinsics, alignment.color_size[0]),
                          alignment.color_size, to_color.tolist())
