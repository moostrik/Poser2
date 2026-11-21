"""`mirrored_image` flips a frame in its own layout, NV12's chroma pairs kept pairs; `bgr_frame` converts an NV12
frame to BGR; `oriented_image` and `oriented_corners` apply the turn and the flips in the contract's order."""

import unittest

import cv2
import numpy as np

from modules.camera import CameraFrame, FrameType, bgr_frame, oriented_corners, oriented_image
from modules.camera.frame import mirrored_image


def _frame(image: np.ndarray, nv12: bool = False, rotate90: bool = False, flip_v: bool = False,
           flip_h: bool = False) -> CameraFrame:
    return CameraFrame(0, FrameType.VIDEO, image, 0.0, nv12=nv12, rotate90=rotate90, flip_v=flip_v, flip_h=flip_h)


# 2 high, 3 wide: each pixel its own value
IMAGE: np.ndarray = np.array([[1, 2, 3],
                              [4, 5, 6]], dtype=np.uint8)


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


class BgrFrameTest(unittest.TestCase):
    def test_a_bgr_or_gray_frame_is_handed_back_as_it_is(self) -> None:
        frame = _frame(IMAGE)
        self.assertIs(bgr_frame(frame), frame)

    def test_an_nv12_frame_becomes_bgr_with_its_flags_kept(self) -> None:
        bgr = np.zeros((4, 6, 3), np.uint8)
        bgr[:, :3] = (255, 0, 0)        # blue on the left, red on the right
        bgr[:, 3:] = (0, 0, 255)
        i420 = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_I420)        # Y, then U and V each a quarter
        y, u, v = i420[:4], i420[4:5].reshape(2, 3), i420[5:6].reshape(2, 3)
        nv12 = np.concatenate((y, np.stack((u, v), axis=-1).reshape(2, 6)))
        converted = bgr_frame(_frame(nv12, nv12=True, rotate90=True, flip_h=True))
        self.assertFalse(converted.nv12)
        self.assertEqual((converted.rotate90, converted.flip_h), (True, True))
        self.assertEqual(converted.image.shape, (4, 6, 3))
        self.assertGreater(int(converted.image[0, 0, 0]), 200)     # blue stays left
        self.assertGreater(int(converted.image[0, 5, 2]), 200)     # red stays right
        self.assertEqual(oriented_image(converted).shape, (6, 4, 3))


class UprightImageTest(unittest.TestCase):
    def test_without_flags_the_image_is_the_frame_s(self) -> None:
        frame = _frame(IMAGE)
        self.assertIs(oriented_image(frame), frame.image)

    def test_the_turn_is_a_transpose(self) -> None:
        np.testing.assert_array_equal(oriented_image(_frame(IMAGE, rotate90=True)), [[1, 4], [2, 5], [3, 6]])

    def test_the_flips_come_after_the_turn(self) -> None:
        np.testing.assert_array_equal(oriented_image(_frame(IMAGE, rotate90=True, flip_v=True)), [[3, 6], [2, 5], [1, 4]])
        np.testing.assert_array_equal(oriented_image(_frame(IMAGE, rotate90=True, flip_h=True)), [[4, 1], [5, 2], [6, 3]])

    def test_a_color_image_keeps_its_channels(self) -> None:
        bgr = np.stack([IMAGE, IMAGE * 10, IMAGE * 20], axis=-1)
        upright = oriented_image(_frame(bgr, rotate90=True, flip_h=True))
        self.assertEqual(upright.shape, (3, 2, 3))
        np.testing.assert_array_equal(upright[0, 0], [4, 40, 80])
        self.assertTrue(upright.flags['C_CONTIGUOUS'])


class UprightCornersTest(unittest.TestCase):
    def test_each_corner_samples_the_pixel_upright_image_shows_there(self) -> None:
        """A corner's source, read from the frame's image, is the upright image's pixel at that corner."""
        for rotate90 in (False, True):
            for flip_v in (False, True):
                for flip_h in (False, True):
                    frame = _frame(IMAGE, rotate90=rotate90, flip_v=flip_v, flip_h=flip_h)
                    upright = oriented_image(frame)
                    quad = oriented_corners(frame)
                    h, w = IMAGE.shape
                    for corner, (row, column) in ((quad.top_left, (0, 0)), (quad.top_right, (0, -1)),
                                                  (quad.bottom_right, (-1, -1)), (quad.bottom_left, (-1, 0))):
                        source = IMAGE[min(int(corner.y * h), h - 1), min(int(corner.x * w), w - 1)]
                        self.assertEqual(source, upright[row, column], (rotate90, flip_v, flip_h))


if __name__ == '__main__':
    unittest.main()
