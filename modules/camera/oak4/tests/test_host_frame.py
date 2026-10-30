"""What the host does for a frame the camera leaves alone: NV12 unpadded, and boxes and keypoints
mirrored with the image. Runs on a depthai 3 install."""

import unittest
from types import SimpleNamespace

import numpy as np

from modules.camera.oak4.definitions import mirrored_keypoints, mirrored_roi, packed_nv12
from modules.utils import Rect


class PackedNv12Test(unittest.TestCase):
    def test_the_padding_comes_off_both_planes(self) -> None:
        width, height, stride, plane_height = 4, 2, 6, 4
        data = np.full(stride * plane_height * 3 // 2, 255, dtype=np.uint8)     # padding is 255 everywhere
        data[:4] = [1, 2, 3, 4]
        data[stride:stride + 4] = [5, 6, 7, 8]
        uv_start: int = stride * plane_height
        data[uv_start:uv_start + 4] = [9, 10, 11, 12]
        frame = SimpleNamespace(getWidth=lambda: width, getHeight=lambda: height, getStride=lambda: stride,
                                getPlaneHeight=lambda: plane_height, getData=lambda: data)
        np.testing.assert_array_equal(packed_nv12(frame), [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])


class MirrorTest(unittest.TestCase):
    def test_a_box_keeps_its_size_and_row(self) -> None:
        roi = mirrored_roi(Rect(0.1, 0.2, 0.3, 0.4))
        self.assertAlmostEqual(roi.x, 0.6)
        self.assertEqual((roi.y, roi.width, roi.height), (0.2, 0.3, 0.4))

    def test_keypoints_flip_x_only_and_leave_the_original(self) -> None:
        points = np.array([[0.25, 0.5, 0.9]], dtype=np.float32)
        np.testing.assert_array_almost_equal(mirrored_keypoints(points), [[0.75, 0.5, 0.9]])
        self.assertEqual(points[0, 0], 0.25)
        self.assertIsNone(mirrored_keypoints(None))


if __name__ == '__main__':
    unittest.main()
