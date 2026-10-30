"""The sensor controls: the panel's sensor fields written to the device, and the device's auto
values read back into them. The OAK-D backend's `SensorControl` for the OAK4's color sensor."""

import logging

import depthai as dai

from .settings import Oak4CameraSettings, SensorSettings

logger = logging.getLogger(__name__)


class SensorControl:
    """Binds one camera's sensor fields for as long as the device is open.

    Bound on construction, when every current value is sent once; `close()` unbinds. The
    callbacks run on the panel's thread and only send; the readback runs on the camera's.
    """

    def __init__(self, settings: Oak4CameraSettings, control: dai.InputQueue) -> None:
        self.settings: Oak4CameraSettings = settings
        self.sensor: SensorSettings = settings.sensor
        self.control: dai.InputQueue = control
        self._bindings: list[tuple] = [
            (SensorSettings.auto_exposure, self._on_auto_exposure),
            (SensorSettings.exposure,      self._on_exposure),
            (SensorSettings.iso,           self._on_exposure),
            (SensorSettings.auto_balance,  self._on_auto_balance),
            (SensorSettings.balance,       self._on_balance),
            (SensorSettings.brightness,    self._on_brightness),
            (SensorSettings.contrast,      self._on_contrast),
            (SensorSettings.saturation,    self._on_saturation),
            (SensorSettings.denoise,       self._on_luma_denoise),
            (SensorSettings.sharpness,     self._on_sharpness),
        ]
        for field, callback in self._bindings:
            self.sensor.bind(field, callback)
        self._apply()

    def close(self) -> None:
        for field, callback in self._bindings:
            self.sensor.unbind(field, callback)
        self._bindings = []

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

    def readback(self, frame: dai.ImgFrame) -> None:
        """What the sensor chose, into the fields the auto modes hold."""
        if self.sensor.auto_exposure:
            self.sensor.exposure = int(frame.getExposureTime().total_seconds() * 1000000)
            self.sensor.iso = frame.getSensitivity()
        if self.sensor.auto_balance:
            self.sensor.balance = frame.getColorTemperature()
