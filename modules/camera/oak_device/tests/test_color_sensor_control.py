"""The color sensor's controls, shared by both OAKs: every value goes out once at the open, a change sends its own,
and the auto values come back off a frame. Runs without a device, on depthai 2 or 3."""

import unittest
from datetime import timedelta
from unittest.mock import MagicMock

import depthai as dai

from modules.camera.oak_device import ColorSensorControl
from modules.settings import BaseSettings, Field


class Sensor(BaseSettings):
    exposure:      Field[int]  = Field(1000)
    iso:           Field[int]  = Field(100)
    auto_exposure: Field[bool] = Field(True)
    balance:       Field[int]  = Field(4000)
    auto_balance:  Field[bool] = Field(True)
    brightness:    Field[int]  = Field(0)
    contrast:      Field[int]  = Field(0)
    saturation:    Field[int]  = Field(0)
    denoise:       Field[int]  = Field(0)
    sharpness:     Field[int]  = Field(0)


class FakeQueue:
    def __init__(self) -> None:
        self.sent: list[dai.CameraControl] = []

    def send(self, msg: dai.CameraControl) -> None:
        self.sent.append(msg)


class ColorSensorControlTest(unittest.TestCase):
    def setUp(self) -> None:
        self.sensor = Sensor()
        self.queue = FakeQueue()
        self.control = ColorSensorControl(self.sensor, self.queue)

    def tearDown(self) -> None:
        self.control.close()

    def test_every_value_goes_out_once_at_the_open(self) -> None:
        # auto exposure, auto balance, brightness, contrast, saturation, luma denoise, sharpness
        self.assertEqual(len(self.queue.sent), 7)

    def test_a_change_sends_its_value(self) -> None:
        self.sensor.auto_exposure = False                   # manual: the exposure goes out
        self.sensor.exposure = 2000
        self.sensor.sharpness = 2
        self.assertEqual(len(self.queue.sent), 10)

    def test_after_closing_a_change_sends_nothing(self) -> None:
        self.control.close()
        self.sensor.contrast = 3
        self.assertEqual(len(self.queue.sent), 7)

    def test_the_auto_values_come_back_off_a_frame(self) -> None:
        frame = MagicMock()
        frame.getExposureTime.return_value = timedelta(microseconds=8333)
        frame.getSensitivity.return_value = 400
        frame.getColorTemperature.return_value = 5600
        self.control.readback(frame)
        self.assertEqual((self.sensor.exposure, self.sensor.iso, self.sensor.balance), (8333, 400, 5600))

    def test_a_manual_value_is_not_read_back(self) -> None:
        self.sensor.auto_exposure = False
        self.sensor.auto_balance = False
        frame = MagicMock()
        frame.getExposureTime.return_value = timedelta(microseconds=8333)
        frame.getSensitivity.return_value = 400
        frame.getColorTemperature.return_value = 5600
        self.control.readback(frame)
        self.assertEqual((self.sensor.exposure, self.sensor.iso, self.sensor.balance), (1000, 100, 4000))


if __name__ == '__main__':
    unittest.main()
