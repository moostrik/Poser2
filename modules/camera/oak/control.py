"""The sensor controls: the panel's sensor fields written to the device, and the device's auto
values read back into them. The role SMILE's camera has under the same name."""

import logging

import depthai as dai

from .settings import OakCameraSettings, ColorSensorSettings, MonoSensorSettings

logger = logging.getLogger(__name__)


class SensorControl:
    """Binds one camera's sensor fields for as long as the device is open.

    A colour camera takes `color_sensor` (exposure, white balance and the image controls); a mono
    one takes `mono_sensor` (exposure and the IR lights, which are device calls rather than
    controls). Bound on construction, when every current value is sent once; `close()` unbinds.
    The callbacks run on the panel's thread and only send; the readback runs on the camera's.
    """

    def __init__(self, settings: OakCameraSettings, device: dai.Device, control: dai.DataInputQueue, color: bool) -> None:
        self.settings: OakCameraSettings = settings
        self.device: dai.Device = device
        self.control: dai.DataInputQueue = control
        self.color: bool = color
        self.sensor: ColorSensorSettings | MonoSensorSettings = settings.color_sensor if color else settings.mono_sensor
        self._bindings: list[tuple] = self._list_bindings()
        for group, field, callback in self._bindings:
            group.bind(field, callback)
        self._apply()

    def close(self) -> None:
        for group, field, callback in self._bindings:
            group.unbind(field, callback)
        self._bindings = []

    def _list_bindings(self) -> list[tuple]:
        if self.color:
            s = self.settings.color_sensor
            return [
                (s, ColorSensorSettings.auto_exposure, self._on_auto_exposure),
                (s, ColorSensorSettings.exposure,      self._on_exposure),
                (s, ColorSensorSettings.iso,           self._on_exposure),
                (s, ColorSensorSettings.auto_balance,  self._on_auto_balance),
                (s, ColorSensorSettings.balance,       self._on_balance),
                (s, ColorSensorSettings.brightness,    self._on_brightness),
                (s, ColorSensorSettings.contrast,      self._on_contrast),
                (s, ColorSensorSettings.saturation,    self._on_saturation),
                (s, ColorSensorSettings.denoise,       self._on_luma_denoise),
                (s, ColorSensorSettings.sharpness,     self._on_sharpness),
            ]
        m = self.settings.mono_sensor
        return [
            (m, MonoSensorSettings.auto_exposure,  self._on_auto_exposure),
            (m, MonoSensorSettings.exposure,       self._on_exposure),
            (m, MonoSensorSettings.iso,            self._on_exposure),
            (m, MonoSensorSettings.ir_flood_light, self._on_ir_flood_light),
            (m, MonoSensorSettings.ir_grid_light,  self._on_ir_grid_light),
        ]

    def _apply(self) -> None:
        """Every current value, once, after the open."""
        self._on_auto_exposure()
        if self.color:
            self._on_auto_balance()
            self._on_brightness()
            self._on_contrast()
            self._on_saturation()
            self._on_luma_denoise()
            self._on_sharpness()
        else:
            self._on_ir_flood_light()
            self._on_ir_grid_light()

    def _send(self, ctrl: dai.CameraControl) -> None:
        try:
            self.control.send(ctrl)
        except Exception as exc:                    # a closing device; the next open applies again
            logger.warning(f'camera control not sent: {exc}')

    # ── Exposure, both sensors ─────────────────────────────────────────

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

    # ── Colour ─────────────────────────────────────────────────────────

    def _on_auto_balance(self, value=None) -> None:
        if self.settings.color_sensor.auto_balance:
            ctrl = dai.CameraControl()
            ctrl.setAutoWhiteBalanceMode(dai.CameraControl.AutoWhiteBalanceMode.AUTO)
            self._send(ctrl)
        else:
            self._on_balance()

    def _on_balance(self, value=None) -> None:
        if self.settings.color_sensor.auto_balance:
            return
        ctrl = dai.CameraControl()
        ctrl.setManualWhiteBalance(self.settings.color_sensor.balance)
        self._send(ctrl)

    def _on_brightness(self, value=None) -> None:
        ctrl = dai.CameraControl()
        ctrl.setBrightness(self.settings.color_sensor.brightness)
        self._send(ctrl)

    def _on_contrast(self, value=None) -> None:
        ctrl = dai.CameraControl()
        ctrl.setContrast(self.settings.color_sensor.contrast)
        self._send(ctrl)

    def _on_saturation(self, value=None) -> None:
        ctrl = dai.CameraControl()
        ctrl.setSaturation(self.settings.color_sensor.saturation)
        self._send(ctrl)

    def _on_luma_denoise(self, value=None) -> None:
        ctrl = dai.CameraControl()
        ctrl.setLumaDenoise(self.settings.color_sensor.denoise)
        self._send(ctrl)

    def _on_sharpness(self, value=None) -> None:
        ctrl = dai.CameraControl()
        ctrl.setSharpness(self.settings.color_sensor.sharpness)
        self._send(ctrl)

    # ── IR, mono ───────────────────────────────────────────────────────

    def _on_ir_flood_light(self, value=None) -> None:
        self.device.setIrFloodLightIntensity(self.settings.mono_sensor.ir_flood_light)

    def _on_ir_grid_light(self, value=None) -> None:
        self.device.setIrLaserDotProjectorIntensity(self.settings.mono_sensor.ir_grid_light)

    # ── Readback, from the frames ──────────────────────────────────────

    def readback(self, frame: dai.ImgFrame) -> None:
        """What the sensor chose, into the fields the auto modes hold."""
        if self.sensor.auto_exposure:
            self.sensor.exposure = int(frame.getExposureTime().total_seconds() * 1000000)
            self.sensor.iso = frame.getSensitivity()
        if self.color and self.settings.color_sensor.auto_balance:
            self.settings.color_sensor.balance = frame.getColorTemperature()
