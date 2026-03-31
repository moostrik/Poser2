from __future__ import annotations
import depthai as dai

from modules.cam.depthcam.Definitions import (
    EXPOSURE_RANGE, ISO_RANGE, BALANCE_RANGE, CONTRAST_RANGE, BRIGHTNESS_RANGE,
    LUMA_DENOISE_RANGE, SATURATION_RANGE, SHARPNESS_RANGE,
    STEREO_DEPTH_RANGE, STEREO_BRIGHTNESS_RANGE, StereoMedianFilterType,
    Input,
)
from modules.cam.depthcam.Pipeline import get_stereo_config
from modules.settings import Settings, Field, Widget


class CoreSettings(Settings):
    """Per-camera runtime settings. Instantiated N times via Child(count=num_cameras)."""

    # Identity
    device_id:      Field[str]   = Field("", access=Field.INIT, description="Camera device ID (MxID)")

    # Pipeline config (shared from parent via Child.share)
    fps:            Field[float] = Field(30.0, min=1.0, max=120.0, access=Field.INIT, visible=False)
    color:          Field[bool]  = Field(True, access=Field.INIT, visible=False)
    square:         Field[bool]  = Field(True, access=Field.INIT, visible=False)
    stereo:         Field[bool]  = Field(False, access=Field.INIT, visible=False)
    yolo:           Field[bool]  = Field(True, access=Field.INIT, visible=False)
    hd_ready:       Field[bool]  = Field(False, access=Field.INIT, visible=False)
    sim_enabled:    Field[bool]  = Field(False, access=Field.INIT, visible=False)
    model_path:     Field[str]   = Field("models", access=Field.INIT, visible=False)

    # Per-camera pipeline settings (INIT — affect pipeline build)
    show_stereo:    Field[bool]  = Field(False, access=Field.INIT, description="Show stereo visualization")
    manual:         Field[bool]  = Field(False, access=Field.INIT, description="Manual camera control")
    flip_h:         Field[bool]  = Field(False, access=Field.INIT, description="Flip horizontal")
    flip_v:         Field[bool]  = Field(False, access=Field.INIT, description="Flip vertical")
    perspective:    Field[float] = Field(0.0, min=-1.0, max=1.0, access=Field.INIT, description="Perspective correction")

    # --- Color controls (WRITE — runtime adjustable) ---
    color_auto_exposure:Field[bool] = Field(True, widget=Widget.switch, description="Auto exposure")
    color_exposure:     Field[int]  = Field(EXPOSURE_RANGE[0], min=EXPOSURE_RANGE[0], max=EXPOSURE_RANGE[1], widget=Widget.slider, description="Exposure (µs)")
    color_iso:          Field[int]  = Field(ISO_RANGE[0], min=ISO_RANGE[0], max=ISO_RANGE[1], widget=Widget.slider, description="ISO")
    color_auto_balance: Field[bool] = Field(True, widget=Widget.switch, description="Auto white balance")
    color_balance:      Field[int]  = Field(BALANCE_RANGE[0], min=BALANCE_RANGE[0], max=BALANCE_RANGE[1], widget=Widget.slider, description="White balance")
    color_brightness:   Field[int]  = Field(0, min=BRIGHTNESS_RANGE[0], max=BRIGHTNESS_RANGE[1], widget=Widget.slider, description="Brightness")
    color_contrast:     Field[int]  = Field(0, min=CONTRAST_RANGE[0], max=CONTRAST_RANGE[1], widget=Widget.slider, description="Contrast")
    color_saturation:   Field[int]  = Field(0, min=SATURATION_RANGE[0], max=SATURATION_RANGE[1], widget=Widget.slider, description="Saturation")
    color_luma_denoise: Field[int]  = Field(0, min=LUMA_DENOISE_RANGE[0], max=LUMA_DENOISE_RANGE[1], widget=Widget.slider, description="Luma denoise")
    color_sharpness:    Field[int]  = Field(0, min=SHARPNESS_RANGE[0], max=SHARPNESS_RANGE[1], widget=Widget.slider, description="Sharpness")

    # --- Mono controls (WRITE) ---
    mono_auto_exposure: Field[bool] = Field(True, widget=Widget.switch, description="Mono auto exposure")
    mono_exposure:      Field[int]  = Field(EXPOSURE_RANGE[0], min=EXPOSURE_RANGE[0], max=EXPOSURE_RANGE[1], widget=Widget.slider, description="Mono exposure (µs)")
    mono_iso:           Field[int]  = Field(ISO_RANGE[0], min=ISO_RANGE[0], max=ISO_RANGE[1], widget=Widget.slider, description="Mono ISO")

    # --- IR controls (WRITE) ---
    ir_grid_light:  Field[float] = Field(0.0, min=0.0, max=1.0, widget=Widget.slider, description="IR grid projector")
    ir_flood_light: Field[float] = Field(0.0, min=0.0, max=1.0, widget=Widget.slider, description="IR flood light")

    # --- Stereo controls (WRITE) ---
    stereo_depth_min:       Field[int]  = Field(STEREO_DEPTH_RANGE[0], min=STEREO_DEPTH_RANGE[0], max=STEREO_DEPTH_RANGE[1], widget=Widget.slider, description="Depth min (mm)")
    stereo_depth_max:       Field[int]  = Field(STEREO_DEPTH_RANGE[1], min=STEREO_DEPTH_RANGE[0], max=STEREO_DEPTH_RANGE[1], widget=Widget.slider, description="Depth max (mm)")
    stereo_brightness_min:  Field[int]  = Field(0, min=STEREO_BRIGHTNESS_RANGE[0], max=STEREO_BRIGHTNESS_RANGE[1], widget=Widget.slider, description="Stereo brightness min")
    stereo_brightness_max:  Field[int]  = Field(STEREO_BRIGHTNESS_RANGE[1], min=STEREO_BRIGHTNESS_RANGE[0], max=STEREO_BRIGHTNESS_RANGE[1], widget=Widget.slider, description="Stereo brightness max")
    stereo_median_filter:   Field[StereoMedianFilterType] = Field(StereoMedianFilterType.OFF, widget=Widget.select, description="Stereo median filter")

    # --- Auto readback (READ — written by Core from camera frames) ---
    actual_color_exposure:  Field[int]  = Field(0, access=Field.READ, description="Actual exposure (µs)")
    actual_color_iso:       Field[int]  = Field(0, access=Field.READ, description="Actual ISO")
    actual_color_balance:   Field[int]  = Field(0, access=Field.READ, description="Actual white balance")
    actual_mono_exposure:   Field[int]  = Field(0, access=Field.READ, description="Actual mono exposure")
    actual_mono_iso:        Field[int]  = Field(0, access=Field.READ, description="Actual mono ISO")

    # --- Status (READ — written by Core) ---
    fps_video:      Field[float] = Field(0.0, access=Field.READ, description="Video FPS")
    tps:            Field[float] = Field(0.0, access=Field.READ, description="Tracker updates/s")
    num_tracklets:  Field[int]   = Field(0, access=Field.READ, description="Active tracklets")

    # ── Hardware connection ────────────────────────────────────────────

    def connect(self, device: dai.Device, inputs: dict[Input, dai.DataInputQueue], do_color: bool) -> None:
        """Bind reactive WRITE fields to DAI hardware commands."""
        self._device = device
        self._inputs = inputs
        self._do_color = do_color
        self._stereo_config: dai.RawStereoDepthConfig = get_stereo_config(do_color)
        self._bind()
        self._apply()

    def disconnect(self) -> None:
        """Unbind all hardware callbacks."""
        self._unbind()
        self._device = None
        self._inputs = {}

    def _bind(self) -> None:
        # Color controls
        self.bind(CoreSettings.color_auto_exposure, self._on_color_auto_exposure)
        self.bind(CoreSettings.color_exposure, self._on_color_exposure)
        self.bind(CoreSettings.color_iso, self._on_color_iso)
        self.bind(CoreSettings.color_auto_balance, self._on_color_auto_balance)
        self.bind(CoreSettings.color_balance, self._on_color_balance)
        self.bind(CoreSettings.color_brightness, self._on_color_brightness)
        self.bind(CoreSettings.color_contrast, self._on_color_contrast)
        self.bind(CoreSettings.color_saturation, self._on_color_saturation)
        self.bind(CoreSettings.color_luma_denoise, self._on_color_luma_denoise)
        self.bind(CoreSettings.color_sharpness, self._on_color_sharpness)

        # Mono controls
        self.bind(CoreSettings.mono_auto_exposure, self._on_mono_auto_exposure)
        self.bind(CoreSettings.mono_exposure, self._on_mono_exposure)
        self.bind(CoreSettings.mono_iso, self._on_mono_iso)

        # IR controls
        self.bind(CoreSettings.ir_flood_light, self._on_ir_flood_light)
        self.bind(CoreSettings.ir_grid_light, self._on_ir_grid_light)

        # Stereo controls
        self.bind(CoreSettings.stereo_depth_min, self._on_stereo_config)
        self.bind(CoreSettings.stereo_depth_max, self._on_stereo_config)
        self.bind(CoreSettings.stereo_brightness_min, self._on_stereo_config)
        self.bind(CoreSettings.stereo_brightness_max, self._on_stereo_config)
        self.bind(CoreSettings.stereo_median_filter, self._on_stereo_config)

    def _unbind(self) -> None:
        self.unbind(CoreSettings.color_auto_exposure, self._on_color_auto_exposure)
        self.unbind(CoreSettings.color_exposure, self._on_color_exposure)
        self.unbind(CoreSettings.color_iso, self._on_color_iso)
        self.unbind(CoreSettings.color_auto_balance, self._on_color_auto_balance)
        self.unbind(CoreSettings.color_balance, self._on_color_balance)
        self.unbind(CoreSettings.color_brightness, self._on_color_brightness)
        self.unbind(CoreSettings.color_contrast, self._on_color_contrast)
        self.unbind(CoreSettings.color_saturation, self._on_color_saturation)
        self.unbind(CoreSettings.color_luma_denoise, self._on_color_luma_denoise)
        self.unbind(CoreSettings.color_sharpness, self._on_color_sharpness)
        self.unbind(CoreSettings.mono_auto_exposure, self._on_mono_auto_exposure)
        self.unbind(CoreSettings.mono_exposure, self._on_mono_exposure)
        self.unbind(CoreSettings.mono_iso, self._on_mono_iso)
        self.unbind(CoreSettings.ir_flood_light, self._on_ir_flood_light)
        self.unbind(CoreSettings.ir_grid_light, self._on_ir_grid_light)
        self.unbind(CoreSettings.stereo_depth_min, self._on_stereo_config)
        self.unbind(CoreSettings.stereo_depth_max, self._on_stereo_config)
        self.unbind(CoreSettings.stereo_brightness_min, self._on_stereo_config)
        self.unbind(CoreSettings.stereo_brightness_max, self._on_stereo_config)
        self.unbind(CoreSettings.stereo_median_filter, self._on_stereo_config)

    def _apply(self) -> None:
        """Push all current field values to hardware (called once after open)."""
        self._on_color_auto_exposure(self.color_auto_exposure)
        if not self.color_auto_exposure:
            self._send_color_exposure_iso(self.color_exposure, self.color_iso)
        self._on_color_auto_balance(self.color_auto_balance)
        if not self.color_auto_balance:
            self._on_color_balance(self.color_balance)
        self._on_color_brightness(self.color_brightness)
        self._on_color_contrast(self.color_contrast)
        self._on_color_saturation(self.color_saturation)
        self._on_color_luma_denoise(self.color_luma_denoise)
        self._on_color_sharpness(self.color_sharpness)
        self._on_mono_auto_exposure(self.mono_auto_exposure)
        if not self.mono_auto_exposure:
            self._send_mono_exposure_iso(self.mono_exposure, self.mono_iso)
        self._on_stereo_config()
        if not self._do_color:
            self._on_ir_flood_light(self.ir_flood_light)
            self._on_ir_grid_light(self.ir_grid_light)

    # ── Helpers ────────────────────────────────────────────────────────

    def _send_control(self, input: Input, control) -> None:
        if self._device is None:
            return
        if input in self._inputs:
            self._inputs[input].send(control)

    # ── Color callbacks ───────────────────────────────────────────────

    def _on_color_auto_exposure(self, value=None) -> None:
        if self._device is None: return
        if self.color_auto_exposure:
            ctrl = dai.CameraControl()
            ctrl.setAutoExposureEnable()
            self._send_control(Input.COLOR_CONTROL, ctrl)
        else:
            self._send_color_exposure_iso(self.color_exposure, self.color_iso)

    def _send_color_exposure_iso(self, exposure: int, iso: int) -> None:
        if self._device is None: return
        ctrl = dai.CameraControl()
        ctrl.setManualExposure(exposure, iso)
        self._send_control(Input.COLOR_CONTROL, ctrl)

    def _on_color_exposure(self, value: int = 0) -> None:
        if not self.color_auto_exposure:
            self._send_color_exposure_iso(self.color_exposure, self.color_iso)

    def _on_color_iso(self, value: int = 0) -> None:
        if not self.color_auto_exposure:
            self._send_color_exposure_iso(self.color_exposure, self.color_iso)

    def _on_color_auto_balance(self, value=None) -> None:
        if self._device is None: return
        if self.color_auto_balance:
            ctrl = dai.CameraControl()
            ctrl.setAutoWhiteBalanceMode(dai.CameraControl.AutoWhiteBalanceMode.AUTO)
            self._send_control(Input.COLOR_CONTROL, ctrl)
        else:
            self._on_color_balance(self.color_balance)

    def _on_color_balance(self, value: int = 0) -> None:
        if self._device is None: return
        ctrl = dai.CameraControl()
        ctrl.setManualWhiteBalance(self.color_balance)
        self._send_control(Input.COLOR_CONTROL, ctrl)

    def _on_color_brightness(self, value: int = 0) -> None:
        if self._device is None: return
        ctrl = dai.CameraControl()
        ctrl.setBrightness(self.color_brightness)
        self._send_control(Input.COLOR_CONTROL, ctrl)

    def _on_color_contrast(self, value: int = 0) -> None:
        if self._device is None: return
        ctrl = dai.CameraControl()
        ctrl.setContrast(self.color_contrast)
        self._send_control(Input.COLOR_CONTROL, ctrl)

    def _on_color_saturation(self, value: int = 0) -> None:
        if self._device is None: return
        ctrl = dai.CameraControl()
        ctrl.setSaturation(self.color_saturation)
        self._send_control(Input.COLOR_CONTROL, ctrl)

    def _on_color_luma_denoise(self, value: int = 0) -> None:
        if self._device is None: return
        ctrl = dai.CameraControl()
        ctrl.setLumaDenoise(self.color_luma_denoise)
        self._send_control(Input.COLOR_CONTROL, ctrl)

    def _on_color_sharpness(self, value: int = 0) -> None:
        if self._device is None: return
        ctrl = dai.CameraControl()
        ctrl.setSharpness(self.color_sharpness)
        self._send_control(Input.COLOR_CONTROL, ctrl)

    # ── Mono callbacks ────────────────────────────────────────────────

    def _on_mono_auto_exposure(self, value=None) -> None:
        if self._device is None: return
        if self.mono_auto_exposure:
            ctrl = dai.CameraControl()
            ctrl.setAutoExposureEnable()
            self._send_control(Input.MONO_CONTROL, ctrl)
        else:
            self._send_mono_exposure_iso(self.mono_exposure, self.mono_iso)

    def _send_mono_exposure_iso(self, exposure: int, iso: int) -> None:
        if self._device is None: return
        ctrl = dai.CameraControl()
        ctrl.setManualExposure(exposure, iso)
        self._send_control(Input.MONO_CONTROL, ctrl)

    def _on_mono_exposure(self, value: int = 0) -> None:
        if not self.mono_auto_exposure:
            self._send_mono_exposure_iso(self.mono_exposure, self.mono_iso)

    def _on_mono_iso(self, value: int = 0) -> None:
        if not self.mono_auto_exposure:
            self._send_mono_exposure_iso(self.mono_exposure, self.mono_iso)

    # ── IR callbacks ──────────────────────────────────────────────────

    def _on_ir_flood_light(self, value: float = 0.0) -> None:
        if self._device is None or self._do_color: return
        self._device.setIrFloodLightIntensity(self.ir_flood_light)

    def _on_ir_grid_light(self, value: float = 0.0) -> None:
        if self._device is None or self._do_color: return
        self._device.setIrLaserDotProjectorIntensity(self.ir_grid_light)

    # ── Stereo callback ───────────────────────────────────────────────

    def _on_stereo_config(self, value=None) -> None:
        if self._device is None: return
        self._stereo_config.postProcessing.thresholdFilter.minRange = self.stereo_depth_min
        self._stereo_config.postProcessing.thresholdFilter.maxRange = self.stereo_depth_max
        self._stereo_config.postProcessing.brightnessFilter.minBrightness = self.stereo_brightness_min
        self._stereo_config.postProcessing.brightnessFilter.maxBrightness = self.stereo_brightness_max
        mf = self.stereo_median_filter
        if mf == StereoMedianFilterType.OFF:
            self._stereo_config.postProcessing.median = dai.MedianFilter.MEDIAN_OFF
        elif mf == StereoMedianFilterType.KERNEL_3x3:
            self._stereo_config.postProcessing.median = dai.MedianFilter.KERNEL_3x3
        elif mf == StereoMedianFilterType.KERNEL_5x5:
            self._stereo_config.postProcessing.median = dai.MedianFilter.KERNEL_5x5
        elif mf == StereoMedianFilterType.KERNEL_7x7:
            self._stereo_config.postProcessing.median = dai.MedianFilter.KERNEL_7x7
        self._send_control(Input.STEREO_CONTROL, self._stereo_config)

    # ── Readback (called by Core from camera frames) ──────────────────

    def update_color_readback(self, frame: dai.ImgFrame) -> None:
        if self.color_auto_exposure:
            self.actual_color_exposure = int(frame.getExposureTime().total_seconds() * 1000000)
            self.actual_color_iso = frame.getSensitivity()
        if self.color_auto_balance:
            self.actual_color_balance = frame.getColorTemperature()

    def update_mono_readback(self, frame: dai.ImgFrame) -> None:
        if self.mono_auto_exposure:
            self.actual_mono_exposure = int(frame.getExposureTime().total_seconds() * 1000000)
            self.actual_mono_iso = frame.getSensitivity()








