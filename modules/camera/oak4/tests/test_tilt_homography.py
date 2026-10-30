"""When the frames are straightened anew: a new homography only for a real change. Runs on a depthai 3
install, without a device."""

import math
import unittest

from modules.camera.oak4 import Camera, Oak4CameraSettings

LENS: list[list[float]] = [[538.0, 0.0, 450.0], [0.0, 538.0, 360.0], [0.0, 0.0, 1.0]]


class UpdateHomographyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = Oak4CameraSettings()
        self.settings.tilt_correction = True
        self.camera = Camera(self.settings)
        self.camera.frame_size = (900, 720)
        self.camera._intrinsics = LENS

    def test_a_small_change_keeps_the_array(self) -> None:
        self.camera._update_homography(-30.0)
        first = self.camera._homography
        self.assertIsNotNone(first)
        self.camera._update_homography(-30.2)
        self.assertIs(self.camera._homography, first)

    def test_a_real_change_makes_a_new_array(self) -> None:
        self.camera._update_homography(-30.0)
        first = self.camera._homography
        self.camera._update_homography(-30.5)
        self.assertIsNot(self.camera._homography, first)

    def test_the_offset_adds_to_the_tilt(self) -> None:
        self.camera._update_homography(-30.0)
        first = self.camera._homography
        self.settings.tilt_offset = 5.0
        self.camera._update_homography(-30.0)
        self.assertIsNot(self.camera._homography, first)
        self.assertAlmostEqual(self.camera._homography_tilt, -25.0)

    def test_no_reading_or_no_correction_is_no_homography(self) -> None:
        self.camera._update_homography(math.nan)
        self.assertIsNone(self.camera._homography)
        self.camera._update_homography(-30.0)
        self.settings.tilt_correction = False
        self.camera._update_homography(-30.0)
        self.assertIsNone(self.camera._homography)


if __name__ == '__main__':
    unittest.main()
