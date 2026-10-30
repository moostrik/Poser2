"""`TiltCorrection`: when the frames are straightened anew, a new homography only for a real change."""

import math
import unittest

from modules.camera import TiltCorrection, TiltCorrectionSettings

LENS: list[list[float]] = [[538.0, 0.0, 450.0], [0.0, 538.0, 360.0], [0.0, 0.0, 1.0]]
SIZE: tuple[int, int] = (900, 720)


class TiltCorrectionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = TiltCorrectionSettings()
        self.settings.enabled = True
        self.correction = TiltCorrection(self.settings)

    def test_a_small_change_keeps_the_array(self) -> None:
        self.correction.update(-30.0, LENS, SIZE)
        first = self.correction.homography
        self.assertIsNotNone(first)
        self.correction.update(-30.2, LENS, SIZE)
        self.assertIs(self.correction.homography, first)

    def test_a_real_change_makes_a_new_array(self) -> None:
        self.correction.update(-30.0, LENS, SIZE)
        first = self.correction.homography
        self.correction.update(-30.5, LENS, SIZE)
        self.assertIsNot(self.correction.homography, first)

    def test_the_offset_adds_to_the_tilt(self) -> None:
        self.correction.update(-30.0, LENS, SIZE)
        first = self.correction.homography
        self.settings.offset = 5.0
        self.correction.update(-30.0, LENS, SIZE)
        self.assertIsNot(self.correction.homography, first)

    def test_no_reading_no_lens_or_no_correction_is_no_homography(self) -> None:
        self.correction.update(math.nan, LENS, SIZE)
        self.assertIsNone(self.correction.homography)
        self.correction.update(-30.0, None, SIZE)
        self.assertIsNone(self.correction.homography)
        self.correction.update(-30.0, LENS, SIZE)
        self.settings.enabled = False
        self.correction.update(-30.0, LENS, SIZE)
        self.assertIsNone(self.correction.homography)


if __name__ == '__main__':
    unittest.main()
