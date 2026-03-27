from modules.cam.depthcam.Definitions import (
    EXPOSURE_RANGE, ISO_RANGE, BALANCE_RANGE, CONTRAST_RANGE, BRIGHTNESS_RANGE,
    LUMA_DENOISE_RANGE, SATURATION_RANGE, SHARPNESS_RANGE,
    STEREO_DEPTH_RANGE, STEREO_BRIGHTNESS_RANGE, StereoMedianFilterType,
)
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












