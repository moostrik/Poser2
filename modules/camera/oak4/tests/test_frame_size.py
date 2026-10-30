"""The delivered frame's shape, and the aligned depth's. Runs on a depthai 3 install."""

import unittest

from modules.camera import CameraResolution
from modules.camera.oak4.definitions import DepthModel, depth_size, direct_size
from modules.camera.oak4.pipeline import MountConfig, delivered_size


def mount(keystone: float = 0.0) -> MountConfig:
    return MountConfig(flip_h=True, flip_v=False, keystone=keystone, fov_h=102.0)


class DeliveredSizeTest(unittest.TestCase):
    def test_the_direct_frame_is_five_by_four(self) -> None:
        self.assertEqual(direct_size(CameraResolution.P720), (900, 720))
        self.assertEqual(delivered_size(True, CameraResolution.P720, mount()), (900, 720))

    def test_a_keystoned_frame_keeps_the_full_width(self) -> None:
        self.assertEqual(delivered_size(True, CameraResolution.P720, mount(keystone=0.2)), (1280, 720))


class DepthSizeTest(unittest.TestCase):
    def test_the_rows_are_the_models_and_the_width_is_even(self) -> None:
        self.assertEqual(depth_size(DepthModel.NEURAL_480, (900, 720)), (376, 300))
        self.assertEqual(depth_size(DepthModel.NEURAL_384, (900, 720)), (300, 240))
        self.assertEqual(depth_size(DepthModel.NEURAL_288, (900, 720)), (224, 180))


if __name__ == '__main__':
    unittest.main()
