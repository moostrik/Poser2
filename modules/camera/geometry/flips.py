"""Boxes and keypoints as they are in the frame flipped left to right (`flip_h`), top to bottom (`flip_v`) and turned
(`rotate90`, transposed)."""

import numpy as np

from modules.utils import Rect


def flip_h_roi(roi: Rect) -> Rect:
    """A normalized box in the frame flipped left to right."""
    return Rect(1.0 - roi.x - roi.width, roi.y, roi.width, roi.height)


def flip_h_keypoints(keypoints: np.ndarray | None) -> np.ndarray | None:
    """Keypoints (N, 3) with x flipped; the labels stay the model's."""
    if keypoints is None:
        return None
    mirrored: np.ndarray = keypoints.copy()
    mirrored[:, 0] = 1.0 - mirrored[:, 0]
    return mirrored


def flip_v_roi(roi: Rect) -> Rect:
    """A normalized box in the frame flipped top to bottom."""
    return Rect(roi.x, 1.0 - roi.y - roi.height, roi.width, roi.height)


def flip_v_keypoints(keypoints: np.ndarray | None) -> np.ndarray | None:
    """Keypoints (N, 3) with y flipped; the labels stay the model's."""
    if keypoints is None:
        return None
    flipped: np.ndarray = keypoints.copy()
    flipped[:, 1] = 1.0 - flipped[:, 1]
    return flipped


def rotate90_roi(roi: Rect) -> Rect:
    """A normalized box in the frame turned as `CameraFrame.rotate90` turns it: transposed."""
    return Rect(roi.y, roi.x, roi.height, roi.width)


def rotate90_keypoints(keypoints: np.ndarray | None) -> np.ndarray | None:
    """Keypoints (N, 3) in the transposed frame: x and y swapped; the labels stay the model's."""
    if keypoints is None:
        return None
    turned: np.ndarray = keypoints.copy()
    turned[:, [0, 1]] = keypoints[:, [1, 0]]
    return turned
