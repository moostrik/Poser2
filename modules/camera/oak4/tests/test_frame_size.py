"""The delivered frame's shape, and the aligned depth's. Runs on a depthai 3 install."""

import unittest

from modules.camera import CameraResolution
from modules.camera.oak4.definitions import FRAME_WIDTH_ALIGNMENT, DepthModel, depth_size, direct_size
from modules.camera.oak4.pipeline import delivered_size


class DeliveredSizeTest(unittest.TestCase):
    def test_the_direct_frame_is_about_five_by_four_on_a_width_the_decoder_plays(self) -> None:
        self.assertEqual(direct_size(CameraResolution.P720), (896, 720))
        self.assertEqual(delivered_size(True, CameraResolution.P720), (896, 720))
        for resolution in (CameraResolution.P720, CameraResolution.P800, CameraResolution.P1080):
            self.assertEqual(direct_size(resolution)[0] % FRAME_WIDTH_ALIGNMENT, 0)

    def test_a_grey_frame_keeps_the_full_width(self) -> None:
        self.assertEqual(delivered_size(False, CameraResolution.P720), (1280, 720))


class DepthSizeTest(unittest.TestCase):
    def test_the_rows_are_the_models_and_the_width_is_even(self) -> None:
        self.assertEqual(depth_size(DepthModel.NEURAL_480, (896, 720)), (374, 300))
        self.assertEqual(depth_size(DepthModel.NEURAL_384, (896, 720)), (298, 240))
        self.assertEqual(depth_size(DepthModel.NEURAL_288, (896, 720)), (224, 180))


if __name__ == '__main__':
    unittest.main()
