"""The correction a camera's settings pick on the OAK4, whose chip warps grey frames only. Runs on a
depthai 3 install."""

import unittest

from modules.camera.oak4.pipeline import Correction, pick_correction


class PickCorrectionTest(unittest.TestCase):
    def test_a_colour_frame_without_tilt_is_cropped_not_refused(self) -> None:
        self.assertEqual(pick_correction(color=True, tilt=0.0, keystone=0.0), Correction.FOUR_POINT)
        self.assertEqual(pick_correction(color=False, tilt=0.0, keystone=0.0), Correction.WARP)
        self.assertEqual(pick_correction(color=True, tilt=0.0, keystone=-0.22), Correction.FOUR_POINT)
        with self.assertRaises(ValueError):
            pick_correction(color=True, tilt=10.0, keystone=0.0)


if __name__ == '__main__':
    unittest.main()
