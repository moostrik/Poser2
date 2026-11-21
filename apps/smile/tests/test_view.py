"""The view: the part of a camera frame the screen shows, as a rectangle (`view_rect`) and as a resize (`cover`)."""

import unittest

import numpy as np

from apps.smile.view import cover, view_rect


class TestView(unittest.TestCase):
    def test_a_frame_of_the_output_shape_is_shown_whole(self):
        view = view_rect(1000, 1250, 1024, 1280)
        self.assertAlmostEqual(view.x, 0.0)
        self.assertAlmostEqual(view.y, 0.0)
        self.assertAlmostEqual(view.width, 1.0)
        self.assertAlmostEqual(view.height, 1.0)

    def test_a_wider_frame_shows_its_middle_columns(self):
        view = view_rect(2000, 1250, 1000, 1250)
        self.assertAlmostEqual(view.x, 0.25)
        self.assertAlmostEqual(view.width, 0.5)
        self.assertAlmostEqual(view.y, 0.0)
        self.assertAlmostEqual(view.height, 1.0)

    def test_a_taller_frame_shows_its_middle_rows(self):
        view = view_rect(1280, 960, 1280, 720)
        self.assertAlmostEqual(view.x, 0.0)
        self.assertAlmostEqual(view.height, 0.75)
        self.assertAlmostEqual(view.y, 0.125)

    def test_cover_shows_what_view_rect_names(self):
        image = np.zeros((40, 80, 3), dtype=np.uint8)
        image[:, 20:60] = 255                               # the middle half of the columns
        covered = cover(image, 20, 20)
        self.assertEqual(covered.shape, (20, 20, 3))
        self.assertTrue(np.all(covered == 255))             # view_rect(80, 40, 20, 20) is x 0.25 to 0.75
        view = view_rect(80, 40, 20, 20)
        self.assertAlmostEqual(view.x, 0.25)
        self.assertAlmostEqual(view.width, 0.5)

    def test_cover_keeps_the_aspect_and_fills_the_size(self):
        wide = np.zeros((10, 40, 3), dtype=np.uint8)
        self.assertEqual(cover(wide, 8, 6).shape, (6, 8, 3))
        tall = np.zeros((40, 10, 3), dtype=np.uint8)
        self.assertEqual(cover(tall, 8, 6).shape, (6, 8, 3))


if __name__ == '__main__':
    unittest.main()
