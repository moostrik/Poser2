"""The NV12 shader's BT.709 video-range matrix, and the plane layout it writes, against numpy."""

import unittest

import numpy as np

from modules.gl import nv12_bytes, shaders


def to_codes(rgb: np.ndarray) -> np.ndarray:
    """Y, Cb, Cr codes 0..255 for RGB 0..255, as the shader computes them and GL rounds them to 8 bits."""
    matrix, offset = shaders.bt709_video_range()
    return np.round((matrix @ (np.asarray(rgb, np.float64) / 255.0) + offset) * 255.0)


def nv12_reference(image: np.ndarray) -> np.ndarray:
    """NV12 of an (h, w, 3) RGB image stored bottom row first, as a GL texture: Y, then Cb Cr at half size, top
    row first, each chroma sample from the average of its 2 × 2 pixels."""
    matrix, offset = shaders.bt709_video_range()
    top_first = image[::-1].astype(np.float64) / 255.0
    height, width, _ = image.shape
    y = np.round((top_first @ matrix[0] + offset[0]) * 255.0)
    mean = top_first.reshape(height // 2, 2, width // 2, 2, 3).mean(axis=(1, 3))
    cb = np.round((mean @ matrix[1] + offset[1]) * 255.0)
    cr = np.round((mean @ matrix[2] + offset[2]) * 255.0)
    uv = np.stack([cb, cr], axis=-1).reshape(height // 2, width)
    return np.concatenate([y, uv]).astype(np.uint8)


class Bt709Test(unittest.TestCase):
    def test_black_and_white_are_the_ends_of_video_range(self) -> None:
        np.testing.assert_array_equal(to_codes([0, 0, 0]), [16, 128, 128])
        np.testing.assert_array_equal(to_codes([255, 255, 255]), [235, 128, 128])

    def test_primaries_match_the_bt709_table(self) -> None:
        np.testing.assert_array_equal(to_codes([255, 0, 0]), [63, 102, 240])
        np.testing.assert_array_equal(to_codes([0, 255, 0]), [173, 42, 26])
        np.testing.assert_array_equal(to_codes([0, 0, 255]), [32, 240, 118])

    def test_chroma_stays_inside_16_to_240(self) -> None:
        corners = np.array([[r, g, b] for r in (0, 255) for g in (0, 255) for b in (0, 255)])
        codes = np.stack([to_codes(c) for c in corners])
        self.assertTrue(((codes[:, 1:] >= 16) & (codes[:, 1:] <= 240)).all())

    def test_matrix_rows_are_luma_and_two_differences(self) -> None:
        matrix, _ = shaders.bt709_video_range()
        self.assertAlmostEqual(float(matrix[0].sum()) * 255.0, 219.0, places=3)
        self.assertAlmostEqual(float(matrix[1].sum()), 0.0, places=6)
        self.assertAlmostEqual(float(matrix[2].sum()), 0.0, places=6)


class Nv12LayoutTest(unittest.TestCase):
    def test_frame_bytes_are_one_and_a_half_per_pixel(self) -> None:
        self.assertEqual(nv12_bytes(3840, 2160), 3840 * 2160 * 3 // 2)

    def test_first_row_is_the_top_and_chroma_is_the_2x2_average(self) -> None:
        image = np.zeros((4, 4, 3), np.uint8)
        image[3] = 255          # the top row on screen: GL stores it last
        frame = nv12_reference(image)
        self.assertEqual(frame.shape, (6, 4))
        np.testing.assert_array_equal(frame[0], [235] * 4)
        np.testing.assert_array_equal(frame[1:4], np.full((3, 4), 16))
        np.testing.assert_array_equal(frame[4:], np.full((2, 4), 128))     # gray stays neutral

    def test_red_block_gives_its_codes_in_its_quarter(self) -> None:
        image = np.zeros((4, 4, 3), np.uint8)
        image[2:, :2] = [255, 0, 0]     # top left on screen
        frame = nv12_reference(image)
        np.testing.assert_array_equal(frame[0:2, 0:2], np.full((2, 2), 63))
        np.testing.assert_array_equal(frame[4, 0:2], [102, 240])
        np.testing.assert_array_equal(frame[4, 2:4], [128, 128])


if __name__ == '__main__':
    unittest.main()
