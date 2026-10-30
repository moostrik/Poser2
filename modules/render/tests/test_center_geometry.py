"""`CenterGeometry` keeps the output's proportions whatever the camera frame's shape."""

import math
import unittest

from modules.render.layers.center.center_geometry import CenterGeometry
from modules.utils import Point2f


OUTPUT_ASPECT: float = 9.0 / 16.0


class CalculateRoiTest(unittest.TestCase):
    def test_a_square_frame_gets_the_plain_crop(self) -> None:
        rotation, distance, roi = CenterGeometry._calculate_roi(
            Point2f(0.5, 0.3), Point2f(0.5, 0.6), Point2f(0.5, 0.3), Point2f(0.5, 0.33), 0.27, OUTPUT_ASPECT)
        self.assertAlmostEqual(rotation, 0.0)
        self.assertAlmostEqual(distance, 0.3)
        self.assertAlmostEqual(roi.height, 0.3 / 0.27)
        self.assertAlmostEqual(roi.width, 0.3 / 0.27 * OUTPUT_ASPECT)

    def test_a_wider_frame_keeps_the_output_aspect_in_pixels(self) -> None:
        cam_aspect: float = 5.0 / 4.0
        _, _, roi = CenterGeometry._calculate_roi(
            Point2f(0.5, 0.3), Point2f(0.5, 0.6), Point2f(0.5, 0.3), Point2f(0.5, 0.33), 0.27, OUTPUT_ASPECT, cam_aspect)
        self.assertAlmostEqual(roi.width * cam_aspect / roi.height, OUTPUT_ASPECT)

    def test_a_lean_is_measured_in_true_proportions(self) -> None:
        cam_aspect: float = 5.0 / 4.0
        rotation, distance, _ = CenterGeometry._calculate_roi(
            Point2f(0.5, 0.3), Point2f(0.5 + 0.1 / cam_aspect, 0.4), Point2f(0.5, 0.3), Point2f(0.5, 0.33), 0.27, OUTPUT_ASPECT, cam_aspect)
        self.assertAlmostEqual(rotation, math.pi / 4.0)
        self.assertAlmostEqual(distance, math.hypot(0.1, 0.1))


if __name__ == '__main__':
    unittest.main()
