"""`fit` places a source of any shape inside a destination, whole, unstretched and centered; `fill` covers it. A
`Rect` fits and fills through them, and gives its corners as a `Quad`."""

import math
import unittest

from modules.utils import Point2f, Quad, Rect, fill, fit


class FitTest(unittest.TestCase):
    def assert_fits(self, src_width: int, src_height: int, dst_width: int, dst_height: int) -> None:
        x, y, width, height = fit(src_width, src_height, dst_width, dst_height)
        self.assertAlmostEqual(width / height, src_width / src_height)
        self.assertLessEqual(width, dst_width)
        self.assertLessEqual(height, dst_height)
        self.assertAlmostEqual(x, dst_width - x - width)
        self.assertAlmostEqual(y, dst_height - y - height)

    def test_fit_shows_all_of_a_wide_source_in_a_tall_target(self) -> None:
        self.assertEqual(fit(200, 100, 100, 100), (0.0, 25.0, 100.0, 50.0))

    def test_fit_shows_all_of_a_tall_source_in_a_wide_target(self) -> None:
        self.assertEqual(fit(100, 200, 100, 100), (25.0, 0.0, 50.0, 100.0))

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

    def test_fill_covers_the_target_and_crops_the_rest(self) -> None:
        self.assertEqual(fill(200, 100, 100, 100), (-50.0, 0.0, 200.0, 100.0))
        self.assertEqual(fill(100, 200, 100, 100), (0.0, -50.0, 100.0, 200.0))

    def test_the_same_shape_fills_the_target_either_way(self) -> None:
        self.assertEqual(fit(16, 9, 1920, 1080), (0.0, 0.0, 1920.0, 1080.0))
        self.assertEqual(fill(16, 9, 1920, 1080), (0.0, 0.0, 1920.0, 1080.0))


class RectTest(unittest.TestCase):
    def test_aspect_fit_is_fit_moved_to_the_target(self) -> None:
        target = Rect(10.0, 20.0, 100.0, 100.0)
        self.assertEqual(Rect(0.0, 0.0, 200.0, 100.0).aspect_fit(target), Rect(10.0, 45.0, 100.0, 50.0))
        self.assertEqual(Rect(0.0, 0.0, 100.0, 200.0).aspect_fit(target), Rect(35.0, 20.0, 50.0, 100.0))

    def test_aspect_fill_is_fill_moved_to_the_target(self) -> None:
        target = Rect(10.0, 20.0, 100.0, 100.0)
        self.assertEqual(Rect(0.0, 0.0, 200.0, 100.0).aspect_fill(target), Rect(-40.0, 20.0, 200.0, 100.0))
        self.assertEqual(Rect(0.0, 0.0, 100.0, 200.0).aspect_fill(target), Rect(10.0, -30.0, 100.0, 200.0))

    def test_intersect_and_union(self) -> None:
        a = Rect(0.0, 0.0, 10.0, 10.0)
        b = Rect(5.0, 5.0, 10.0, 10.0)
        self.assertEqual(a.intersect(b), Rect(5.0, 5.0, 5.0, 5.0))
        self.assertEqual(a.union(b), Rect(0.0, 0.0, 15.0, 15.0))
        self.assertTrue(a.intersect(Rect(20.0, 20.0, 1.0, 1.0)).is_empty)

    def test_corners_go_round_from_the_top_left(self) -> None:
        self.assertEqual(Rect(1.0, 2.0, 3.0, 4.0).corners(),
                         Quad(Point2f(1.0, 2.0), Point2f(4.0, 2.0), Point2f(4.0, 6.0), Point2f(1.0, 6.0)))

    def test_a_quarter_turn_moves_each_corner_one_place(self) -> None:
        rect = Rect(0.25, 0.25, 0.5, 0.5)
        turned = rect.turned(math.pi / 2, rect.center)
        corners = rect.corners()
        for moved, place in ((turned.top_left, corners.bottom_left), (turned.top_right, corners.top_left),
                             (turned.bottom_right, corners.top_right), (turned.bottom_left, corners.bottom_right)):
            self.assertAlmostEqual(moved.x, place.x)
            self.assertAlmostEqual(moved.y, place.y)


if __name__ == '__main__':
    unittest.main()
