"""`mirrored_image` flips a frame in its own layout; NV12's chroma pairs must stay pairs."""

import unittest

import numpy as np

from modules.camera import CameraFrame, FrameType, mirrored_image


def _frame(image: np.ndarray, nv12: bool = False) -> CameraFrame:
    return CameraFrame(0, FrameType.VIDEO, image, 0.0, nv12=nv12)


class MirroredImageTest(unittest.TestCase):
    def test_bgr_and_gray_flip_their_columns(self) -> None:
        gray = np.array([[1, 2], [3, 4]], dtype=np.uint8)
        np.testing.assert_array_equal(mirrored_image(_frame(gray)), [[2, 1], [4, 3]])
        bgr = np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.uint8)
        np.testing.assert_array_equal(mirrored_image(_frame(bgr)), [[[4, 5, 6], [1, 2, 3]]])

    def test_nv12_keeps_each_uv_pair_in_order(self) -> None:
        image = np.array([[1, 2, 3, 4],
                          [5, 6, 7, 8],
                          [10, 20, 30, 40]], dtype=np.uint8)
        np.testing.assert_array_equal(mirrored_image(_frame(image, nv12=True)),
                                      [[4, 3, 2, 1],
                                       [8, 7, 6, 5],
                                       [30, 40, 10, 20]])


if __name__ == '__main__':
    unittest.main()
