"""The color sensor's controls, the same on both OAKs: its fields written to the device, its auto values read back
off the frames, on depthai 2 and 3 alike."""

import logging
from collections.abc import Callable
from typing import Any, Protocol

import depthai as dai

from modules.settings import Field

logger = logging.getLogger(__name__)


class ColorSensorFields(Protocol):
    """A color sensor's settings as the control binds them: read on the settings each is its value, read on their
    class the `Field` to bind."""
    exposure:      Field[int]
    iso:           Field[int]
    auto_exposure: Field[bool]
    balance:       Field[int]
    auto_balance:  Field[bool]
    brightness:    Field[int]
    contrast:      Field[int]
    saturation:    Field[int]
    denoise:       Field[int]
    sharpness:     Field[int]

    def bind(self, field: Field[Any], callback: Callable[..., None]) -> None: ...
    def unbind(self, field: Field[Any], callback: Callable[..., None]) -> None: ...


class ControlQueue(Protocol):
    """The host's end of a sensor's control input, depthai 2's or 3's."""
    def send(self, msg: dai.CameraControl) -> None: ...


class ColorSensorControl:
    """One color sensor's fields bound to its open device's control queue: every value sent once on construction,
    the auto values read back off each frame, `close()` unbinding."""

    def __init__(self, sensor: ColorSensorFields, control: ControlQueue) -> None:
        self.sensor: ColorSensorFields = sensor
        self.control: ControlQueue = control
        fields = type(sensor)
        self._bindings: list[tuple[Field[Any], Callable[..., None]]] = [
            (fields.auto_exposure, self._on_auto_exposure),
            (fields.exposure,      self._on_exposure),
            (fields.iso,           self._on_exposure),
            (fields.auto_balance,  self._on_auto_balance),
            (fields.balance,       self._on_balance),
            (fields.brightness,    self._on_brightness),
            (fields.contrast,      self._on_contrast),
            (fields.saturation,    self._on_saturation),
            (fields.denoise,       self._on_luma_denoise),
            (fields.sharpness,     self._on_sharpness),
        ]
        for field, callback in self._bindings:
            sensor.bind(field, callback)
        self._apply()

    def close(self) -> None:
        for field, callback in self._bindings:
            self.sensor.unbind(field, callback)
        self._bindings = []

    def readback(self, frame: dai.ImgFrame) -> None:
        """What the sensor chose, into the fields the auto modes hold."""
        if self.sensor.auto_exposure:
            self.sensor.exposure = int(frame.getExposureTime().total_seconds() * 1000000)
            self.sensor.iso = frame.getSensitivity()
        if self.sensor.auto_balance:
            self.sensor.balance = frame.getColorTemperature()

    def _apply(self) -> None:
        """Every current value, once, after the open."""
        self._on_auto_exposure()
        self._on_auto_balance()
        self._on_brightness()
        self._on_contrast()
        self._on_saturation()
        self._on_luma_denoise()
        self._on_sharpness()

    def _send(self, ctrl: dai.CameraControl) -> None:
        try:
            self.control.send(ctrl)
        except Exception as exc:                    # a closing device; the next open applies again
            logger.warning(f'camera control not sent: {exc}')

    def _on_auto_exposure(self, value=None) -> None:
        if self.sensor.auto_exposure:
            ctrl = dai.CameraControl()
            ctrl.setAutoExposureEnable()
            self._send(ctrl)
        else:
            self._on_exposure()

    def _on_exposure(self, value=None) -> None:
        if self.sensor.auto_exposure:
            return
        ctrl = dai.CameraControl()
        ctrl.setManualExposure(self.sensor.exposure, self.sensor.iso)
        self._send(ctrl)

    def _on_auto_balance(self, value=None) -> None:
        if self.sensor.auto_balance:
            ctrl = dai.CameraControl()
            ctrl.setAutoWhiteBalanceMode(dai.CameraControl.AutoWhiteBalanceMode.AUTO)
            self._send(ctrl)
        else:
            self._on_balance()

    def _on_balance(self, value=None) -> None:
        if self.sensor.auto_balance:
            return
        ctrl = dai.CameraControl()
        ctrl.setManualWhiteBalance(self.sensor.balance)
        self._send(ctrl)

    def _on_brightness(self, value=None) -> None:
        ctrl = dai.CameraControl()
        ctrl.setBrightness(self.sensor.brightness)
        self._send(ctrl)

    def _on_contrast(self, value=None) -> None:
        ctrl = dai.CameraControl()
        ctrl.setContrast(self.sensor.contrast)
        self._send(ctrl)

    def _on_saturation(self, value=None) -> None:
        ctrl = dai.CameraControl()
        ctrl.setSaturation(self.sensor.saturation)
        self._send(ctrl)

    def _on_luma_denoise(self, value=None) -> None:
        ctrl = dai.CameraControl()
        ctrl.setLumaDenoise(self.sensor.denoise)
        self._send(ctrl)

    def _on_sharpness(self, value=None) -> None:
        ctrl = dai.CameraControl()
        ctrl.setSharpness(self.sensor.sharpness)
        self._send(ctrl)
