"""The mono sensor's controls: its exposure and the IR lights written to the device, its auto values read back. The
color sensor's are `oak_device`'s `ColorSensorControl`."""

import logging

import depthai as dai

from .settings import MonoSensorSettings

logger = logging.getLogger(__name__)


class MonoSensorControl:
    """The mono sensor's fields bound to its open device: exposure, ISO, auto exposure and the IR lights; every value
    is sent once on construction, `close()` unbinds."""

    def __init__(self, sensor: MonoSensorSettings, device: dai.Device, control: dai.DataInputQueue) -> None:
        self.sensor: MonoSensorSettings = sensor
        self.device: dai.Device = device
        self.control: dai.DataInputQueue = control
        self._bindings = [
            (MonoSensorSettings.auto_exposure,  self._on_auto_exposure),
            (MonoSensorSettings.exposure,       self._on_exposure),
            (MonoSensorSettings.iso,            self._on_exposure),
            (MonoSensorSettings.flood_light, self._on_flood_light),
            (MonoSensorSettings.dot_projector,  self._on_dot_projector),
        ]
        for field, callback in self._bindings:
            sensor.bind(field, callback)
        self._on_auto_exposure()
        self._on_flood_light()
        self._on_dot_projector()

    def close(self) -> None:
        for field, callback in self._bindings:
            self.sensor.unbind(field, callback)
        self._bindings = []

    def readback(self, frame: dai.ImgFrame) -> None:
        """What the sensor chose, into the fields the auto mode holds."""
        if self.sensor.auto_exposure:
            self.sensor.exposure = int(frame.getExposureTime().total_seconds() * 1000000)
            self.sensor.iso = frame.getSensitivity()

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

    def _on_flood_light(self, value=None) -> None:
        self.device.setIrFloodLightIntensity(self.sensor.flood_light)

    def _on_dot_projector(self, value=None) -> None:
        self.device.setIrLaserDotProjectorIntensity(self.sensor.dot_projector)
