"""One OAK camera's settings: the contract's `CameraSettings` plus what shapes its pipeline and
the sensor controls. Pure data; `SensorControl` writes the sensor fields to the device."""

from .. import CameraSettings, CameraResolution
from modules.settings import BaseSettings, Field, Group, Widget

from .definitions import (
    EXPOSURE_RANGE, ISO_RANGE, BALANCE_RANGE, CONTRAST_RANGE, BRIGHTNESS_RANGE,
    LUMA_DENOISE_RANGE, SATURATION_RANGE, SHARPNESS_RANGE,
)


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


class OakCameraSettings(CameraSettings):
    """One OAK camera: the contract's rate, mount and readings, plus what shapes its pipeline.

    The tunables live in subgroups — `readings`, `color_sensor`, `mono_sensor` — because forty
    flat fields in one panel is not something anyone reads. What stays at this level is what
    shapes the pipeline (`color`, `square`, `yolo`, `resolution`) and the mount (`CameraSettings`),
    which are the fields a person actually sets. The preset keys are flat: the same `cam_0.tilt`
    as every other backend.
    """

    # Initial settings
    device_id:      Field[str]   = Field("", access=Field.INIT, description="Camera device ID (MxID)")
    color:          Field[bool]  = Field(True, access=Field.INIT, newline=True, description="Color camera (False = mono)")
    square:         Field[bool]  = Field(True, access=Field.INIT)
    yolo:           Field[bool]  = Field(True, access=Field.INIT)
    resolution:     Field[CameraResolution] = Field(CameraResolution.P800, access=Field.INIT, description="Sensor mode. P1080 is colour only; mono falls back to P800")
    model_path:     Field[str]   = Field("data/models", access=Field.INIT)

    # Two values arrive shared from the camera group and are relayed one level further into
    # `mono_sensor`, so the panel shows them where the rest of the mono controls are rather than
    # stranded at this level. Invisible here: they are plumbing, not a second copy to set.
    mono_auto_exposure: Field[bool]  = Field(True, visible=False, description="Relayed to mono_sensor.auto_exposure")
    ir_flood_light:     Field[float] = Field(0.0, min=0.0, max=1.0, visible=False, description="Relayed to mono_sensor.ir_flood_light")

    color_sensor: Group[ColorSensorSettings] = Group(ColorSensorSettings)
    mono_sensor:  Group[MonoSensorSettings]  = Group(MonoSensorSettings, share=[mono_auto_exposure.as_('auto_exposure'), ir_flood_light])
