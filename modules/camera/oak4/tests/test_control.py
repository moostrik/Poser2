"""What the live controls send, without a device: the depth config, and the exposure region in the
sensor's pixels. Runs on a depthai 3 install."""

import unittest

from modules.camera.oak4.control import DepthControl
from modules.camera.oak4.definitions import ANTI_BANDING_MODES, FAR_LIMIT_M, NO_FAR_LIMIT_MM, AntiBanding, sensor_region
from modules.camera.oak4.settings import DepthSettings, Oak4CameraSettings
from modules.utils import Rect


class FakeQueue:
    def __init__(self) -> None:
        self.sent: list = []

    def send(self, message) -> None:
        self.sent.append(message)


class DepthControlTest(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = DepthSettings()
        self.queue = FakeQueue()
        self.control = DepthControl(self.settings, self.queue)

    def tearDown(self) -> None:
        self.control.close()

    def test_the_values_go_out_once_at_the_open(self) -> None:
        self.assertEqual(len(self.queue.sent), 1)
        self.assertEqual(self.queue.sent[0].getConfidenceThreshold(), 125)
        self.assertEqual(self.queue.sent[0].getEdgeThreshold(), 10)
        self.assertFalse(self.queue.sent[0].postProcessing.temporalFilter.enable)

    def test_a_change_sends_every_value(self) -> None:
        self.settings.confidence = 200
        self.settings.temporal_filter = True
        self.settings.temporal_alpha = 0.25
        self.assertEqual(len(self.queue.sent), 4)
        message = self.queue.sent[-1]
        self.assertEqual(message.getConfidenceThreshold(), 200)
        self.assertTrue(message.postProcessing.temporalFilter.enable)
        self.assertAlmostEqual(message.postProcessing.temporalFilter.alpha, 0.25)

    def test_near_and_far_go_out_in_millimetres(self) -> None:
        self.settings.near = 0.5
        self.settings.far = 4.0
        limits = self.queue.sent[-1].postProcessing.thresholdFilter
        self.assertEqual((limits.minRange, limits.maxRange), (500, 4000))

    def test_far_at_the_top_is_no_limit(self) -> None:
        self.settings.far = 4.0
        self.settings.far = FAR_LIMIT_M
        self.assertEqual(self.queue.sent[-1].postProcessing.thresholdFilter.maxRange, NO_FAR_LIMIT_MM)

    def test_after_closing_a_change_sends_nothing(self) -> None:
        self.control.close()
        self.settings.confidence = 90
        self.assertEqual(len(self.queue.sent), 1)


class SensorRegionTest(unittest.TestCase):
    @staticmethod
    def to_source(x: float, y: float) -> tuple[float, float]:
        """A frame that is the sensor's middle: 4000x3000 with 1000 cut off each side, none off the top."""
        return 1000.0 + x * 2000.0, y * 3000.0

    def test_the_whole_picture_is_the_cropped_part_of_the_sensor(self) -> None:
        self.assertEqual(sensor_region(Rect(0.0, 0.0, 1.0, 1.0), False, self.to_source), (1000, 0, 2000, 3000))

    def test_a_region_lands_in_the_sensors_pixels(self) -> None:
        self.assertEqual(sensor_region(Rect(0.25, 0.5, 0.5, 0.25), False, self.to_source), (1500, 1500, 1000, 750))

    def test_the_picture_is_mirrored_after_the_camera_so_the_region_is_mirrored_back(self) -> None:
        self.assertEqual(sensor_region(Rect(0.0, 0.0, 0.25, 1.0), True, self.to_source), (2500, 0, 500, 3000))


class SettingsTest(unittest.TestCase):
    def test_every_anti_banding_mode_has_its_depthai_mode(self) -> None:
        self.assertEqual(set(ANTI_BANDING_MODES), set(AntiBanding))

    def test_the_region_and_the_depth_group_survive_a_preset(self) -> None:
        settings = Oak4CameraSettings()
        settings.sensor.exposure_region = Rect(0.1, 0.2, 0.3, 0.4)
        settings.depth.far = 5.0
        saved = settings.to_dict()
        loaded = Oak4CameraSettings()
        loaded.update_from_dict(saved)
        self.assertEqual(loaded.sensor.exposure_region, Rect(0.1, 0.2, 0.3, 0.4))
        self.assertEqual(loaded.depth.far, 5.0)


if __name__ == '__main__':
    unittest.main()
