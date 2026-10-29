"""Every shared resolution label maps to a depthai sensor mode on both sensors."""

import unittest

from modules.camera import CameraResolution
from modules.camera.oak.definitions import mono_mode, color_mode


class SensorModeTest(unittest.TestCase):
    def test_shared_labels_map_to_both_sensors(self) -> None:
        for label in (CameraResolution.P720, CameraResolution.P800):
            self.assertIsNotNone(mono_mode(label))
            self.assertIsNotNone(color_mode(label))

    def test_mono_falls_back_from_a_colour_only_label(self) -> None:
        self.assertEqual(mono_mode(CameraResolution.P1080), mono_mode(CameraResolution.P800))


if __name__ == '__main__':
    unittest.main()
