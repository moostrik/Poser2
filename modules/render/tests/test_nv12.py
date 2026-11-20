"""The NV12 shader's BT.709 video-range matrix against the BT.709 table, and the frame's bytes; the shader's own
output is checked on the GPU (scripts/screen_capture_check.py)."""

import unittest

import numpy as np

from modules.render import nv12_bytes
from modules.render.general.screen_recording import shaders


def to_codes(rgb: list[int]) -> np.ndarray:
    """Y, Cb, Cr codes 0..255 for RGB 0..255 by the matrix the shader gets, rounded as GL rounds to 8 bits."""
    matrix, offset = shaders.bt709_video_range()
    return np.round((matrix @ (np.asarray(rgb, np.float64) / 255.0) + offset) * 255.0)


class Bt709Test(unittest.TestCase):
    def test_black_and_white_are_the_ends_of_video_range(self) -> None:
        np.testing.assert_array_equal(to_codes([0, 0, 0]), [16, 128, 128])
        np.testing.assert_array_equal(to_codes([255, 255, 255]), [235, 128, 128])

    def test_primaries_match_the_bt709_table(self) -> None:
        np.testing.assert_array_equal(to_codes([255, 0, 0]), [63, 102, 240])
        np.testing.assert_array_equal(to_codes([0, 255, 0]), [173, 42, 26])
        np.testing.assert_array_equal(to_codes([0, 0, 255]), [32, 240, 118])

    def test_chroma_stays_inside_16_to_240(self) -> None:
        corners = [[r, g, b] for r in (0, 255) for g in (0, 255) for b in (0, 255)]
        codes = np.stack([to_codes(c) for c in corners])
        self.assertTrue(((codes[:, 1:] >= 16) & (codes[:, 1:] <= 240)).all())

    def test_matrix_rows_are_luma_and_two_differences(self) -> None:
        matrix, _ = shaders.bt709_video_range()
        self.assertAlmostEqual(float(matrix[0].sum()) * 255.0, 219.0, places=3)
        self.assertAlmostEqual(float(matrix[1].sum()), 0.0, places=6)
        self.assertAlmostEqual(float(matrix[2].sum()), 0.0, places=6)


class Nv12BytesTest(unittest.TestCase):
    def test_frame_bytes_are_one_and_a_half_per_pixel(self) -> None:
        self.assertEqual(nv12_bytes(3840, 2160), 3840 * 2160 * 3 // 2)


if __name__ == '__main__':
    unittest.main()
