"""One OAK4 camera's settings: the contract's `RigCameraSettings` plus what shapes its pipeline and
the sensor controls. Pure data; `SensorControl` writes the sensor fields to the device."""

from .. import RigCameraSettings, CameraResolution
from modules.settings import BaseSettings, Field, Group, Widget

from .definitions import (
    PoseChain, CameraLink, DepthModel, EXPOSURE_RANGE, ISO_RANGE, BALANCE_RANGE, CONTRAST_RANGE, BRIGHTNESS_RANGE,
    LUMA_DENOISE_RANGE, SATURATION_RANGE, SHARPNESS_RANGE,
)


class SensorSettings(BaseSettings):
    """The color sensor's exposure, balance and image controls, which apply whether the camera
    delivers color or grey, and the OAK4 Pro's two IR lights, which only the stereo pair sees.
    The stereo pair's exposure is its own auto exposure."""
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
    dot_projector:  Field[float] = Field(0.0, min=0.0, max=1.0, step=0.05, widget=Widget.slider, description="IR dot projector brightness", newline=True)
    flood_light:    Field[float] = Field(0.0, min=0.0, max=1.0, step=0.05, widget=Widget.slider, description="IR flood light brightness")


class Oak4CameraSettings(RigCameraSettings):
    """One OAK4 camera: the contract's rate, mount and readings, plus what shapes its pipeline.

    The same fields at this level as the OAK-D's `OakCameraSettings`, so an app shares them the
    same way (`fps`, `color`, `yolo`, `resolution`, `model_path`, `fov`) and the preset keys stay
    flat. `color` picks the delivered frame, color or grey, off the color sensor.
    """
    device_id:      Field[str]   = Field("", access=Field.INIT, description="Camera device ID (serial, or its IP; empty = the first OAK4 found)")
    link:           Field[CameraLink] = Field(CameraLink.USB, access=Field.INIT, description="Link to the PC; the other one is used when this one is missing")
    color:          Field[bool]  = Field(True, access=Field.INIT, newline=True, description="Color frames (False = grey off the same sensor)")
    yolo:           Field[bool]  = Field(True, access=Field.INIT)
    pose_chain:     Field[PoseChain] = Field(PoseChain.ONE_MODEL, access=Field.INIT, description="Models on the camera: one pose model, a detector and a pose model on each crop, or a detector alone")
    depth_model:    Field[DepthModel] = Field(DepthModel.OFF, access=Field.INIT, description="Neural Depth size on the camera; larger is sharper and slower")
    resolution:     Field[CameraResolution] = Field(CameraResolution.P800, access=Field.INIT, description="Delivered size before the crop; the sensor mode follows it")
    model_path:     Field[str]   = Field("data/models", access=Field.INIT)
    tilt_correction: Field[bool]  = Field(False, widget=Widget.switch, newline=True, description="Straighten color and depth for the camera's tilt")
    tilt_offset:    Field[float] = Field(0.0, min=-60.0, max=60.0, step=0.5, widget=Widget.slider, description="Added to the IMU's tilt (°)")

    sensor: Group[SensorSettings] = Group(SensorSettings)
