"""The camera's live controls: the panel's sensor fields written to the device with the device's auto
values read back into them (the OAK-D backend's `SensorControl` for the OAK4's color sensor, and the
two IR lights), and the depth fields written to Neural Depth."""

import logging
from typing import Callable

import depthai as dai

from .definitions import ANTI_BANDING_MODES, FAR_LIMIT_M, NO_FAR_LIMIT_MM, sensor_region
from .settings import DepthSettings, Oak4CameraSettings, SensorSettings

logger = logging.getLogger(__name__)


class SensorControl:
    """Binds one camera's sensor fields for as long as the device is open.

    Bound on construction, when every current value is sent once; `close()` unbinds and turns the
    IR lights off, as they would stay lit. The callbacks run on the panel's thread and only send;
    the readback runs on the camera's. The exposure region waits for the first frame, the only
    place the sensor's pixels can be read from.
    """

    def __init__(self, settings: Oak4CameraSettings, device: dai.Device, control: dai.InputQueue, flip_h: bool = False) -> None:
        self.settings: Oak4CameraSettings = settings
        self.sensor: SensorSettings = settings.sensor
        self.device: dai.Device = device
        self.control: dai.InputQueue = control
        self.flip_h: bool = flip_h
        self._transformation: dai.ImgTransformation | None = None
        self._bindings: list[tuple] = [
            (SensorSettings.auto_exposure, self._on_auto_exposure),
            (SensorSettings.exposure,      self._on_exposure),
            (SensorSettings.iso,           self._on_exposure),
            (SensorSettings.auto_exposure_lock, self._on_exposure_lock),
            (SensorSettings.exposure_compensation, self._on_exposure_compensation),
            (SensorSettings.auto_iso_limit, self._on_auto_iso_limit),
            (SensorSettings.exposure_region, self._on_exposure_region),
            (SensorSettings.anti_banding,  self._on_anti_banding),
            (SensorSettings.auto_balance,  self._on_auto_balance),
            (SensorSettings.auto_balance_lock, self._on_balance_lock),
            (SensorSettings.balance,       self._on_balance),
            (SensorSettings.brightness,    self._on_brightness),
            (SensorSettings.contrast,      self._on_contrast),
            (SensorSettings.saturation,    self._on_saturation),
            (SensorSettings.denoise,       self._on_luma_denoise),
            (SensorSettings.chroma_denoise, self._on_chroma_denoise),
            (SensorSettings.sharpness,     self._on_sharpness),
            (SensorSettings.dot_projector, self._on_dot_projector),
            (SensorSettings.flood_light,   self._on_flood_light),
        ]
        for field, callback in self._bindings:
            self.sensor.bind(field, callback)
        self._apply()

    def close(self) -> None:
        for field, callback in self._bindings:
            self.sensor.unbind(field, callback)
        self._bindings = []
        self._set_light(self.device.setIrLaserDotProjectorIntensity, 0.0)
        self._set_light(self.device.setIrFloodLightIntensity, 0.0)

    def _apply(self) -> None:
        """Every current value, once, after the open."""
        self._on_anti_banding()
        self._on_exposure_compensation()
        self._on_auto_iso_limit()
        self._on_auto_exposure()
        self._on_auto_balance()
        self._on_brightness()
        self._on_contrast()
        self._on_saturation()
        self._on_luma_denoise()
        self._on_chroma_denoise()
        self._on_sharpness()
        self._on_dot_projector()
        self._on_flood_light()

    def _on_dot_projector(self, value=None) -> None:
        self._set_light(self.device.setIrLaserDotProjectorIntensity, self.sensor.dot_projector)

    def _on_flood_light(self, value=None) -> None:
        self._set_light(self.device.setIrFloodLightIntensity, self.sensor.flood_light)

    def _set_light(self, setter: Callable[[float], object], intensity: float) -> None:
        try:
            setter(intensity)
        except Exception as exc:                    # a closing device, or one without the light
            logger.warning(f'IR light not set: {exc}')

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
            self._on_exposure_lock()
        else:
            self._on_exposure()

    def _on_exposure(self, value=None) -> None:
        if self.sensor.auto_exposure:
            return
        ctrl = dai.CameraControl()
        ctrl.setManualExposure(self.sensor.exposure, self.sensor.iso)
        self._send(ctrl)

    def _on_exposure_lock(self, value=None) -> None:
        ctrl = dai.CameraControl()
        ctrl.setAutoExposureLock(self.sensor.auto_exposure_lock)
        self._send(ctrl)

    def _on_exposure_compensation(self, value=None) -> None:
        ctrl = dai.CameraControl()
        ctrl.setAutoExposureCompensation(self.sensor.exposure_compensation)
        self._send(ctrl)

    def _on_auto_iso_limit(self, value=None) -> None:
        ctrl = dai.CameraControl()
        ctrl.setAutoExposureMaxISO(self.sensor.auto_iso_limit)
        self._send(ctrl)

    def _on_exposure_region(self, value=None) -> None:
        transformation: dai.ImgTransformation | None = self._transformation
        if transformation is None:
            return
        width, height = transformation.getSize()

        def to_source(x: float, y: float) -> tuple[float, float]:
            point = transformation.invTransformPoint(dai.Point2f(x * width, y * height, False))
            return point.x, point.y
        ctrl = dai.CameraControl()
        ctrl.setAutoExposureRegion(*sensor_region(self.sensor.exposure_region, self.flip_h, to_source))
        self._send(ctrl)

    def _on_anti_banding(self, value=None) -> None:
        ctrl = dai.CameraControl()
        ctrl.setAntiBandingMode(ANTI_BANDING_MODES[self.sensor.anti_banding])
        self._send(ctrl)

    def _on_auto_balance(self, value=None) -> None:
        if self.sensor.auto_balance:
            ctrl = dai.CameraControl()
            ctrl.setAutoWhiteBalanceMode(dai.CameraControl.AutoWhiteBalanceMode.AUTO)
            self._send(ctrl)
            self._on_balance_lock()
        else:
            self._on_balance()

    def _on_balance_lock(self, value=None) -> None:
        ctrl = dai.CameraControl()
        ctrl.setAutoWhiteBalanceLock(self.sensor.auto_balance_lock)
        self._send(ctrl)

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

    def _on_chroma_denoise(self, value=None) -> None:
        ctrl = dai.CameraControl()
        ctrl.setChromaDenoise(self.sensor.chroma_denoise)
        self._send(ctrl)

    def _on_sharpness(self, value=None) -> None:
        ctrl = dai.CameraControl()
        ctrl.setSharpness(self.sensor.sharpness)
        self._send(ctrl)

    def readback(self, frame: dai.ImgFrame) -> None:
        """What the sensor chose, into the fields the auto modes hold."""
        if self._transformation is None:
            self._transformation = frame.getTransformation()
            self._on_exposure_region()
        if self.sensor.auto_exposure:
            self.sensor.exposure = int(frame.getExposureTime().total_seconds() * 1000000)
            self.sensor.iso = frame.getSensitivity()
        if self.sensor.auto_balance:
            self.sensor.balance = frame.getColorTemperature()


class DepthControl:
    """Binds one camera's depth fields for as long as the device is open: every change sends all of
    them in one message, which is what Neural Depth takes. The callbacks run on the panel's thread."""

    def __init__(self, settings: DepthSettings, queue: dai.InputQueue) -> None:
        self.settings: DepthSettings = settings
        self.queue: dai.InputQueue = queue
        self._fields: list = [DepthSettings.confidence, DepthSettings.edge, DepthSettings.temporal_filter, DepthSettings.temporal_alpha,
                              DepthSettings.temporal_delta, DepthSettings.near, DepthSettings.far]
        for field in self._fields:
            self.settings.bind(field, self._send)
        self._send()

    def close(self) -> None:
        for field in self._fields:
            self.settings.unbind(field, self._send)
        self._fields = []

    def build_message(self) -> dai.NeuralDepthConfig:
        s: DepthSettings = self.settings
        message = dai.NeuralDepthConfig()
        message.setConfidenceThreshold(s.confidence)
        message.setEdgeThreshold(s.edge)
        message.postProcessing.temporalFilter.enable = s.temporal_filter
        message.postProcessing.temporalFilter.alpha = s.temporal_alpha
        message.postProcessing.temporalFilter.delta = s.temporal_delta
        message.postProcessing.thresholdFilter.minRange = round(s.near * 1000.0)
        message.postProcessing.thresholdFilter.maxRange = NO_FAR_LIMIT_MM if s.far >= FAR_LIMIT_M else round(s.far * 1000.0)
        return message

    def _send(self, value=None) -> None:
        try:
            self.queue.send(self.build_message())
        except Exception as exc:                    # a closing device; the next open applies again
            logger.warning(f'depth settings not sent: {exc}')
