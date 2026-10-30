"""The camera's own report of its load reaches the readings, and leaves them when it closes. Runs
without a device. On a depthai 3 install (`.venv-oak4`)."""

import unittest
from types import SimpleNamespace
from typing import Any

from modules.camera.oak4 import Oak4CameraSettings
from modules.camera.oak4.camera import Camera


class SystemReadingsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.camera = Camera(Oak4CameraSettings())
        message: Any = SimpleNamespace(cpuAvgUsage=SimpleNamespace(average=0.48), ddrMemoryUsage=SimpleNamespace(used=1, total=4),
                                       chipTemperature=SimpleNamespace(average=54.5))
        self.camera._system_callback('system', message)

    def test_the_report_fills_the_readings_in_percent_and_degrees(self) -> None:
        readings = self.camera.settings.readings
        self.assertAlmostEqual(readings.cpu, 48.0)
        self.assertAlmostEqual(readings.memory, 25.0)
        self.assertAlmostEqual(readings.temperature, 54.5)

    def test_a_closed_camera_reports_no_load(self) -> None:
        self.camera._close()
        readings = self.camera.settings.readings
        self.assertEqual((readings.cpu, readings.memory, readings.temperature), (0.0, 0.0, 0.0))


if __name__ == '__main__':
    unittest.main()
