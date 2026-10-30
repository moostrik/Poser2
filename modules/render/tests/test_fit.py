"""`fit` places a source of any shape inside a destination: whole, unstretched and centered."""

import unittest

from modules.gl import fit


class FitTest(unittest.TestCase):
    def assert_fits(self, src_width: int, src_height: int, dst_width: int, dst_height: int) -> None:
        x, y, width, height = fit(src_width, src_height, dst_width, dst_height)
        self.assertAlmostEqual(width / height, src_width / src_height)
        self.assertLessEqual(width, dst_width)
        self.assertLessEqual(height, dst_height)
        self.assertAlmostEqual(x, dst_width - x - width)
        self.assertAlmostEqual(y, dst_height - y - height)

    def test_wide_destination_has_bars_left_and_right(self) -> None:
        self.assert_fits(1080, 1920, 1920, 1080)
        self.assertEqual(fit(1080, 1920, 1920, 1080), (656.25, 0.0, 607.5, 1080))

    def test_narrow_destination_has_bars_top_and_bottom(self) -> None:
        self.assert_fits(1080, 1920, 540, 1960)
        self.assertEqual(fit(1080, 1920, 540, 1960), (0.0, 500.0, 540, 960))

    def test_equal_shape_fills_the_destination(self) -> None:
        self.assert_fits(1080, 1920, 540, 960)
        self.assertEqual(fit(1080, 1920, 540, 960), (0.0, 0.0, 540, 960))

    def test_square_source_in_a_portrait_output(self) -> None:
        self.assert_fits(720, 720, 1080, 1920)


if __name__ == '__main__':
    unittest.main()
