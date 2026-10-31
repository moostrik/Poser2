"""The camera's own report of its load reaches the readings, and leaves them when it closes. Runs
without a device. On a depthai 3 install (`.venv-oak4`)."""

import unittest
from types import SimpleNamespace
from typing import Any

import depthai as dai

from modules.camera.oak4 import Oak4CameraSettings
from modules.camera.oak4.camera import Camera
from modules.camera.oak4.definitions import link_name


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

    def test_a_closed_camera_shows_no_link(self) -> None:
        self.camera.settings.readings.link = 'USB'
        self.camera._close()
        self.assertEqual(self.camera.settings.readings.link, '')


class LinkNameTest(unittest.TestCase):
    def test_the_two_routes_read_as_their_links(self) -> None:
        self.assertEqual(link_name(dai.XLinkProtocol.X_LINK_USB_EP), 'USB')
        self.assertEqual(link_name(dai.XLinkProtocol.X_LINK_TCP_IP), 'ETHERNET')

    def test_another_route_keeps_its_own_name(self) -> None:
        self.assertEqual(link_name(dai.XLinkProtocol.X_LINK_USB_VSC), 'X_LINK_USB_VSC')


if __name__ == '__main__':
    unittest.main()
