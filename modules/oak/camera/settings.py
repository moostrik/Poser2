import depthai as dai

from .definitions import (
    EXPOSURE_RANGE, ISO_RANGE, BALANCE_RANGE, CONTRAST_RANGE, BRIGHTNESS_RANGE,
    LUMA_DENOISE_RANGE, SATURATION_RANGE, SHARPNESS_RANGE,
    STEREO_DEPTH_RANGE, STEREO_BRIGHTNESS_RANGE, StereoMedianFilterType,
    CameraResolution, Input,
)
from .pipeline import get_stereo_config
from modules.settings import BaseSettings, Field, Group, Widget


class CameraCheckSettings(BaseSettings):
    """Pinned verdicts saying whether the cameras are mounted and running the way the preset assumes.

    The per-camera readings live on each `CameraSettings`, groups deep in the panel, which is
    where a readout goes to be ignored. The pinned indicators say good or bad; which camera and
    which value is drawn on that camera's own view in the renderer.
    """
    mount_tolerance: Field[float] = Field(2.0, min=0.5, max=10.0, step=0.5,
                                          description="Tilt or roll deviation (°) a camera may have before the mount warns")
    mount:           Field[bool]  = Field(True, access=Field.READ, pinned=True, widget=Widget.status,
                                          description="Camera tilt and roll within tolerance; each camera view shows its error")
    camera_fps:      Field[bool]  = Field(False, access=Field.READ, pinned=True, widget=Widget.status,
                                          description="Camera frame rates within 5 % of fps; each camera view shows its rate")


class CameraReadings(BaseSettings):
    """What the camera reports back about itself, and the one number that corrects it.

    Nothing in the show reads any of this — it exists for whoever is setting the rig up. All of it
    is read-only except `roll_offset`, which lives here rather than with the mount constants
    because it is the thing you reach for while looking at the roll it corrects.
    """
    video_fps:      Field[float] = Field(0.0, access=Field.READ, description="Video FPS")
    tracker_fps:    Field[float] = Field(0.0, access=Field.READ, description="Tracker updates/s")
    tracklets:      Field[int]   = Field(0, access=Field.READ, description="Active tracklets")
    # How the camera is actually mounted, as the board reports it — NaN until measured, which is
    # the honest default for a board with no IMU.
    tilt_measured:  Field[float] = Field(float('nan'), access=Field.READ, description="Up-tilt (°) from the camera's own IMU — compare with `tilt`")
    roll_measured:  Field[float] = Field(float('nan'), access=Field.READ, description="Roll (°) about the optical axis, `roll_offset` subtracted. Should be 0")
    # No min/max: `Widget.resolve` sends a bounded float to a slider and an unbounded one to a
    # number box, and this is typed in or filled by the zero button, never dragged.
    roll_offset:    Field[float] = Field(0.0, description="What this camera reads when level (°). Corrects the reading, not the image")
    fov_factory:    Field[float] = Field(float('nan'), access=Field.READ, description="Horizontal field (°) this unit's own calibration spans across the frame")
    lens_error:     Field[float] = Field(float('nan'), access=Field.READ, description="Largest bearing error (°) this unit has under the shared lens")


class ColorSensorSettings(BaseSettings):
    """The colour sensor's exposure, balance and image controls. Inert on a mono camera."""
    exposure:       Field[int]  = Field(EXPOSURE_RANGE[0], min=EXPOSURE_RANGE[0], max=EXPOSURE_RANGE[1], access=Field.READWRITE, widget=Widget.slider, description="Exposure (µs)", newline=True)
    iso:            Field[int]  = Field(ISO_RANGE[0], min=ISO_RANGE[0], max=ISO_RANGE[1], access=Field.READWRITE, widget=Widget.slider, description="ISO")
    auto_exposure:  Field[bool] = Field(True, widget=Widget.switch, description="Auto exposure")
    balance:        Field[int]  = Field(BALANCE_RANGE[0], min=BALANCE_RANGE[0], max=BALANCE_RANGE[1], access=Field.READWRITE, widget=Widget.slider, description="White balance", newline=True)
    auto_balance:   Field[bool] = Field(True, widget=Widget.switch, description="Auto white balance")
    brightness:     Field[int]  = Field(0, min=BRIGHTNESS_RANGE[0], max=BRIGHTNESS_RANGE[1], widget=Widget.slider, description="Brightness", newline=True)
    contrast:       Field[int]  = Field(0, min=CONTRAST_RANGE[0], max=CONTRAST_RANGE[1], widget=Widget.slider, description="Contrast")
    saturation:     Field[int]  = Field(0, min=SATURATION_RANGE[0], max=SATURATION_RANGE[1], widget=Widget.slider, description="Saturation")
    denoise:        Field[int]  = Field(0, min=LUMA_DENOISE_RANGE[0], max=LUMA_DENOISE_RANGE[1], widget=Widget.slider, description="Luma denoise")
    sharpness:      Field[int]  = Field(0, min=SHARPNESS_RANGE[0], max=SHARPNESS_RANGE[1], widget=Widget.slider, description="Sharpness")


class MonoSensorSettings(BaseSettings):
    """The mono sensor's exposure, and the IR illumination that goes with it — the flood lamp and
    the dot projector are only ever used by a mono camera, and are ignored on a colour one."""
    exposure:       Field[int]   = Field(EXPOSURE_RANGE[0], min=EXPOSURE_RANGE[0], max=EXPOSURE_RANGE[1], access=Field.READWRITE, widget=Widget.slider, description="Mono exposure (µs)", newline=True)
    iso:            Field[int]   = Field(ISO_RANGE[0], min=ISO_RANGE[0], max=ISO_RANGE[1], access=Field.READWRITE, widget=Widget.slider, description="Mono ISO")
    auto_exposure:  Field[bool]  = Field(True, widget=Widget.switch, description="Mono auto exposure")
    ir_grid_light:  Field[float] = Field(0.0, min=0.0, max=1.0, widget=Widget.slider, description="IR dot projector", newline=True)
    ir_flood_light: Field[float] = Field(0.0, min=0.0, max=1.0, widget=Widget.slider, description="IR flood light")


class DepthSettings(BaseSettings):
    """The stereo depth pair. Unused by every installation so far — no app consumes a depth
    frame, and every preset runs `stereo = false` — so this is kept collapsed and out of the way
    rather than deleted, in case a future rig wants the pair."""
    show:           Field[bool] = Field(False, access=Field.INIT, description="Show the stereo depth visualization")
    depth_min:      Field[int]  = Field(STEREO_DEPTH_RANGE[0], min=STEREO_DEPTH_RANGE[0], max=STEREO_DEPTH_RANGE[1], widget=Widget.slider, description="Depth min (mm)", newline=True)
    depth_max:      Field[int]  = Field(STEREO_DEPTH_RANGE[1], min=STEREO_DEPTH_RANGE[0], max=STEREO_DEPTH_RANGE[1], widget=Widget.slider, description="Depth max (mm)")
    median_filter:  Field[StereoMedianFilterType] = Field(StereoMedianFilterType.OFF, widget=Widget.select, description="Stereo median filter")
    bright_min:     Field[int]  = Field(0, min=STEREO_BRIGHTNESS_RANGE[0], max=STEREO_BRIGHTNESS_RANGE[1], widget=Widget.slider, description="Stereo brightness min", newline=True)
    bright_max:     Field[int]  = Field(STEREO_BRIGHTNESS_RANGE[1], min=STEREO_BRIGHTNESS_RANGE[0], max=STEREO_BRIGHTNESS_RANGE[1], widget=Widget.slider, description="Stereo brightness max")


class CameraSettings(BaseSettings):
    """Per-camera runtime settings. Instantiated N times via Child(count=num_cameras).

    The tunables live in four subgroups — `readings`, `color_sensor`, `mono_sensor`, `depth` —
    because forty flat fields in one panel is not something anyone reads. What stays at this level
    is what shapes the pipeline (`color`, `square`, `stereo`, `yolo`, `resolution`) or describes
    the mount (`flip_*`, `tilt`, `keystone`, `fov`), which are the fields a person actually sets.
    """

    # Initial settings
    device_id:      Field[str]   = Field("", access=Field.INIT, description="Camera device ID (MxID)")
    fps:            Field[float] = Field(30.0, min=1.0, max=120.0, access=Field.INIT, description="Camera FPS")
    color:          Field[bool]  = Field(True, access=Field.INIT, newline=True, description="Color camera (False = mono)")
    square:         Field[bool]  = Field(True, access=Field.INIT)
    stereo:         Field[bool]  = Field(False, access=Field.INIT)
    yolo:           Field[bool]  = Field(True, access=Field.INIT)
    resolution:     Field[CameraResolution] = Field(CameraResolution.P800, access=Field.INIT, description="Sensor mode. P1080 is colour only; mono falls back to P800")
    model_path:     Field[str]   = Field("data/models", access=Field.INIT)
    flip_h:         Field[bool]  = Field(False, access=Field.INIT, description="Flip horizontal")
    flip_v:         Field[bool]  = Field(False, access=Field.INIT, description="Flip vertical")
    tilt:           Field[float] = Field(0.0, access=Field.INIT,
                                         description="Camera up-tilt (°), positive = aimed up. Exclusive with keystone")
    keystone:       Field[float] = Field(0.0, access=Field.INIT,
                                         description="Keystone (fraction of frame). Exclusive with tilt")
    # Relayed down from the camera group so the warp can turn `tilt` into pixels.
    fov:            Field[float] = Field(127.0, access=Field.INIT,
                                         description="Azimuth span (°) of the delivered frame, quoted for the full sensor width")
    # The lens itself — see the lens geometry notes in `definitions.py`. Shared by an
    # installation's cameras; 0 / 0 / 0 is the old model, the lens taken to be `fov`.
    lens_fov:       Field[float] = Field(0.0, access=Field.INIT, step=0.1,
                                         description="Field (°) the lens spans across the full sensor width; 0 = same as fov")
    lens_centre_x:  Field[float] = Field(0.0, access=Field.INIT, step=0.5,
                                         description="Optical centre offset from the frame centre (px), full sensor mode")
    lens_centre_y:  Field[float] = Field(0.0, access=Field.INIT, step=0.5,
                                         description="Optical centre offset from the frame centre (px), positive = down")

    # Two values arrive shared from the camera group and are relayed one level further into
    # `mono_sensor`, so the panel shows them where the rest of the mono controls are rather than
    # stranded at this level. Invisible here: they are plumbing, not a second copy to set.
    mono_auto_exposure: Field[bool]  = Field(True, visible=False, description="Relayed to mono_sensor.auto_exposure")
    ir_flood_light:     Field[float] = Field(0.0, min=0.0, max=1.0, visible=False, description="Relayed to mono_sensor.ir_flood_light")

    readings:     Group[CameraReadings]     = Group(CameraReadings)
    color_sensor: Group[ColorSensorSettings] = Group(ColorSensorSettings)
    mono_sensor:  Group[MonoSensorSettings]  = Group(MonoSensorSettings, share=[mono_auto_exposure.as_('auto_exposure'), ir_flood_light])
    depth:        Group[DepthSettings]       = Group(DepthSettings)

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
        self.color_sensor.bind(ColorSensorSettings.auto_exposure, self._on_color_auto_exposure)
        self.color_sensor.bind(ColorSensorSettings.exposure, self._on_color_exposure)
        self.color_sensor.bind(ColorSensorSettings.iso, self._on_color_iso)
        self.color_sensor.bind(ColorSensorSettings.auto_balance, self._on_color_auto_balance)
        self.color_sensor.bind(ColorSensorSettings.balance, self._on_color_balance)
        self.color_sensor.bind(ColorSensorSettings.brightness, self._on_color_brightness)
        self.color_sensor.bind(ColorSensorSettings.contrast, self._on_color_contrast)
        self.color_sensor.bind(ColorSensorSettings.saturation, self._on_color_saturation)
        self.color_sensor.bind(ColorSensorSettings.denoise, self._on_color_luma_denoise)
        self.color_sensor.bind(ColorSensorSettings.sharpness, self._on_color_sharpness)

        # Mono controls
        self.mono_sensor.bind(MonoSensorSettings.auto_exposure, self._on_mono_auto_exposure)
        self.mono_sensor.bind(MonoSensorSettings.exposure, self._on_mono_exposure)
        self.mono_sensor.bind(MonoSensorSettings.iso, self._on_mono_iso)

        # IR controls
        self.mono_sensor.bind(MonoSensorSettings.ir_flood_light, self._on_ir_flood_light)
        self.mono_sensor.bind(MonoSensorSettings.ir_grid_light, self._on_ir_grid_light)

        # Stereo controls
        self.depth.bind(DepthSettings.depth_min, self._on_stereo_config)
        self.depth.bind(DepthSettings.depth_max, self._on_stereo_config)
        self.depth.bind(DepthSettings.bright_min, self._on_stereo_config)
        self.depth.bind(DepthSettings.bright_max, self._on_stereo_config)
        self.depth.bind(DepthSettings.median_filter, self._on_stereo_config)

    def _unbind(self) -> None:
        self.color_sensor.unbind(ColorSensorSettings.auto_exposure, self._on_color_auto_exposure)
        self.color_sensor.unbind(ColorSensorSettings.exposure, self._on_color_exposure)
        self.color_sensor.unbind(ColorSensorSettings.iso, self._on_color_iso)
        self.color_sensor.unbind(ColorSensorSettings.auto_balance, self._on_color_auto_balance)
        self.color_sensor.unbind(ColorSensorSettings.balance, self._on_color_balance)
        self.color_sensor.unbind(ColorSensorSettings.brightness, self._on_color_brightness)
        self.color_sensor.unbind(ColorSensorSettings.contrast, self._on_color_contrast)
        self.color_sensor.unbind(ColorSensorSettings.saturation, self._on_color_saturation)
        self.color_sensor.unbind(ColorSensorSettings.denoise, self._on_color_luma_denoise)
        self.color_sensor.unbind(ColorSensorSettings.sharpness, self._on_color_sharpness)
        self.mono_sensor.unbind(MonoSensorSettings.auto_exposure, self._on_mono_auto_exposure)
        self.mono_sensor.unbind(MonoSensorSettings.exposure, self._on_mono_exposure)
        self.mono_sensor.unbind(MonoSensorSettings.iso, self._on_mono_iso)
        self.mono_sensor.unbind(MonoSensorSettings.ir_flood_light, self._on_ir_flood_light)
        self.mono_sensor.unbind(MonoSensorSettings.ir_grid_light, self._on_ir_grid_light)
        self.depth.unbind(DepthSettings.depth_min, self._on_stereo_config)
        self.depth.unbind(DepthSettings.depth_max, self._on_stereo_config)
        self.depth.unbind(DepthSettings.bright_min, self._on_stereo_config)
        self.depth.unbind(DepthSettings.bright_max, self._on_stereo_config)
        self.depth.unbind(DepthSettings.median_filter, self._on_stereo_config)

    def _apply(self) -> None:
        """Push all current field values to hardware (called once after open)."""
        self._on_color_auto_exposure(self.color_sensor.auto_exposure)
        if not self.color_sensor.auto_exposure:
            self._send_color_exposure_iso(self.color_sensor.exposure, self.color_sensor.iso)
        self._on_color_auto_balance(self.color_sensor.auto_balance)
        if not self.color_sensor.auto_balance:
            self._on_color_balance(self.color_sensor.balance)
        self._on_color_brightness(self.color_sensor.brightness)
        self._on_color_contrast(self.color_sensor.contrast)
        self._on_color_saturation(self.color_sensor.saturation)
        self._on_color_luma_denoise(self.color_sensor.denoise)
        self._on_color_sharpness(self.color_sensor.sharpness)
        self._on_mono_auto_exposure(self.mono_sensor.auto_exposure)
        if not self.mono_sensor.auto_exposure:
            self._send_mono_exposure_iso(self.mono_sensor.exposure, self.mono_sensor.iso)
        self._on_stereo_config()
        if not self._do_color:
            self._on_ir_flood_light(self.mono_sensor.ir_flood_light)
            self._on_ir_grid_light(self.mono_sensor.ir_grid_light)

    # ── Helpers ────────────────────────────────────────────────────────

    def _send_control(self, input: Input, control) -> None:
        if self._device is None:
            return
        if input in self._inputs:
            self._inputs[input].send(control)

    # ── Color callbacks ───────────────────────────────────────────────

    def _on_color_auto_exposure(self, value=None) -> None:
        if self._device is None: return
        if self.color_sensor.auto_exposure:
            ctrl = dai.CameraControl()
            ctrl.setAutoExposureEnable()
            self._send_control(Input.COLOR_CONTROL, ctrl)
        else:
            self._send_color_exposure_iso(self.color_sensor.exposure, self.color_sensor.iso)

    def _send_color_exposure_iso(self, exposure: int, iso: int) -> None:
        if self._device is None: return
        ctrl = dai.CameraControl()
        ctrl.setManualExposure(exposure, iso)
        self._send_control(Input.COLOR_CONTROL, ctrl)

    def _on_color_exposure(self, value: int = 0) -> None:
        if not self.color_sensor.auto_exposure:
            self._send_color_exposure_iso(self.color_sensor.exposure, self.color_sensor.iso)

    def _on_color_iso(self, value: int = 0) -> None:
        if not self.color_sensor.auto_exposure:
            self._send_color_exposure_iso(self.color_sensor.exposure, self.color_sensor.iso)

    def _on_color_auto_balance(self, value=None) -> None:
        if self._device is None: return
        if self.color_sensor.auto_balance:
            ctrl = dai.CameraControl()
            ctrl.setAutoWhiteBalanceMode(dai.CameraControl.AutoWhiteBalanceMode.AUTO)
            self._send_control(Input.COLOR_CONTROL, ctrl)
        else:
            self._on_color_balance(self.color_sensor.balance)

    def _on_color_balance(self, value: int = 0) -> None:
        if self._device is None or self.color_sensor.auto_balance: return
        ctrl = dai.CameraControl()
        ctrl.setManualWhiteBalance(self.color_sensor.balance)
        self._send_control(Input.COLOR_CONTROL, ctrl)

    def _on_color_brightness(self, value: int = 0) -> None:
        if self._device is None: return
        ctrl = dai.CameraControl()
        ctrl.setBrightness(self.color_sensor.brightness)
        self._send_control(Input.COLOR_CONTROL, ctrl)

    def _on_color_contrast(self, value: int = 0) -> None:
        if self._device is None: return
        ctrl = dai.CameraControl()
        ctrl.setContrast(self.color_sensor.contrast)
        self._send_control(Input.COLOR_CONTROL, ctrl)

    def _on_color_saturation(self, value: int = 0) -> None:
        if self._device is None: return
        ctrl = dai.CameraControl()
        ctrl.setSaturation(self.color_sensor.saturation)
        self._send_control(Input.COLOR_CONTROL, ctrl)

    def _on_color_luma_denoise(self, value: int = 0) -> None:
        if self._device is None: return
        ctrl = dai.CameraControl()
        ctrl.setLumaDenoise(self.color_sensor.denoise)
        self._send_control(Input.COLOR_CONTROL, ctrl)

    def _on_color_sharpness(self, value: int = 0) -> None:
        if self._device is None: return
        ctrl = dai.CameraControl()
        ctrl.setSharpness(self.color_sensor.sharpness)
        self._send_control(Input.COLOR_CONTROL, ctrl)

    # ── Mono callbacks ────────────────────────────────────────────────

    def _on_mono_auto_exposure(self, value=None) -> None:
        if self._device is None: return
        if self.mono_sensor.auto_exposure:
            ctrl = dai.CameraControl()
            ctrl.setAutoExposureEnable()
            self._send_control(Input.MONO_CONTROL, ctrl)
        else:
            self._send_mono_exposure_iso(self.mono_sensor.exposure, self.mono_sensor.iso)

    def _send_mono_exposure_iso(self, exposure: int, iso: int) -> None:
        if self._device is None: return
        ctrl = dai.CameraControl()
        ctrl.setManualExposure(exposure, iso)
        self._send_control(Input.MONO_CONTROL, ctrl)

    def _on_mono_exposure(self, value: int = 0) -> None:
        if not self.mono_sensor.auto_exposure:
            self._send_mono_exposure_iso(self.mono_sensor.exposure, self.mono_sensor.iso)

    def _on_mono_iso(self, value: int = 0) -> None:
        if not self.mono_sensor.auto_exposure:
            self._send_mono_exposure_iso(self.mono_sensor.exposure, self.mono_sensor.iso)

    # ── IR callbacks ──────────────────────────────────────────────────

    def _on_ir_flood_light(self, value: float = 0.0) -> None:
        if self._device is None or self._do_color: return
        self._device.setIrFloodLightIntensity(self.mono_sensor.ir_flood_light)

    def _on_ir_grid_light(self, value: float = 0.0) -> None:
        if self._device is None or self._do_color: return
        self._device.setIrLaserDotProjectorIntensity(self.mono_sensor.ir_grid_light)

    # ── Stereo callback ───────────────────────────────────────────────

    def _on_stereo_config(self, value=None) -> None:
        if self._device is None: return
        self._stereo_config.postProcessing.thresholdFilter.minRange = self.depth.depth_min
        self._stereo_config.postProcessing.thresholdFilter.maxRange = self.depth.depth_max
        self._stereo_config.postProcessing.brightnessFilter.minBrightness = self.depth.bright_min
        self._stereo_config.postProcessing.brightnessFilter.maxBrightness = self.depth.bright_max
        mf = self.depth.median_filter
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
        if self.color_sensor.auto_exposure:
            self.color_sensor.exposure = int(frame.getExposureTime().total_seconds() * 1000000)
            self.color_sensor.iso = frame.getSensitivity()
        if self.color_sensor.auto_balance:
            self.color_sensor.balance = frame.getColorTemperature()

    def update_mono_readback(self, frame: dai.ImgFrame) -> None:
        if self.mono_sensor.auto_exposure:
            self.mono_sensor.exposure = int(frame.getExposureTime().total_seconds() * 1000000)
            self.mono_sensor.iso = frame.getSensitivity()








