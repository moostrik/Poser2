"""The delivered frame's shape. Runs on a depthai 3 install."""

import unittest

from modules.camera import CameraResolution
from modules.camera.oak4.definitions import FRAME_WIDTH_ALIGNMENT, direct_size
from modules.camera.oak4.pipeline import delivered_size


class DeliveredSizeTest(unittest.TestCase):
    def test_the_direct_frame_is_about_five_by_four_on_a_width_the_decoder_plays(self) -> None:
        self.assertEqual(direct_size(CameraResolution.P720), (896, 720))
        self.assertEqual(delivered_size(True, CameraResolution.P720), (896, 720))
        for resolution in (CameraResolution.P720, CameraResolution.P800, CameraResolution.P1080):
            self.assertEqual(direct_size(resolution)[0] % FRAME_WIDTH_ALIGNMENT, 0)

    def test_a_grey_frame_keeps_the_full_width(self) -> None:
        self.assertEqual(delivered_size(False, CameraResolution.P720), (1280, 720))


if __name__ == '__main__':
    unittest.main()
