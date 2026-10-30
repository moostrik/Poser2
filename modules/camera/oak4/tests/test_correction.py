"""The correction a camera's settings pick on the OAK4, whose chip warps grey frames only. Runs on a
depthai 3 install."""

import unittest

from modules.camera.oak4.pipeline import Correction, pick_correction


class PickCorrectionTest(unittest.TestCase):
    def test_grey_is_warped_and_keystone_is_four_point(self) -> None:
        self.assertEqual(pick_correction(color=False, keystone=0.0, flip_v=False), Correction.WARP)
        self.assertEqual(pick_correction(color=True, keystone=-0.22, flip_v=False), Correction.FOUR_POINT)

    def test_a_plain_colour_frame_goes_direct_unless_it_needs_a_vertical_flip(self) -> None:
        self.assertEqual(pick_correction(color=True, keystone=0.0, flip_v=False), Correction.DIRECT)
        self.assertEqual(pick_correction(color=True, keystone=0.0, flip_v=True), Correction.FOUR_POINT)


if __name__ == '__main__':
    unittest.main()
