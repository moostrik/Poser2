"""One OAK4 camera's settings: the contract's `RigCameraSettings` plus what shapes its pipeline, the
sensor controls and the depth controls. Pure data; `SensorControl` and `DepthControl` write them to
the device."""

from .. import RigCameraSettings, CameraResolution, TiltCorrectionSettings
from modules.settings import BaseSettings, Field, Group, Widget
from modules.utils import Rect

from .definitions import (
    ModelChain, CameraLink, DepthModel, AntiBanding, EXPOSURE_RANGE, ISO_RANGE, BALANCE_RANGE, CONTRAST_RANGE, BRIGHTNESS_RANGE,
    LUMA_DENOISE_RANGE, CHROMA_DENOISE_RANGE, SATURATION_RANGE, SHARPNESS_RANGE, COMPENSATION_RANGE, DEPTH_SCORE_RANGE, FAR_LIMIT_M,
)


class SensorSettings(BaseSettings):
    """The color sensor's exposure, balance and image controls, which apply whether the camera
    delivers color or grey, and the OAK4 Pro's two IR lights, which only the stereo pair sees.
    The stereo pair's exposure is its own auto exposure."""
    exposure:       Field[int]  = Field(EXPOSURE_RANGE[0], min=EXPOSURE_RANGE[0], max=EXPOSURE_RANGE[1], access=Field.READWRITE, widget=Widget.slider, description="Exposure (µs)", newline=True)
    iso:            Field[int]  = Field(ISO_RANGE[0], min=ISO_RANGE[0], max=ISO_RANGE[1], access=Field.READWRITE, widget=Widget.slider, description="ISO")
    auto_exposure:  Field[bool] = Field(True, widget=Widget.switch, description="Auto exposure")
    auto_exposure_lock: Field[bool] = Field(False, widget=Widget.switch, description="Hold auto exposure where it is now")
    exposure_compensation: Field[int] = Field(0, min=COMPENSATION_RANGE[0], max=COMPENSATION_RANGE[1], widget=Widget.slider, description="Auto exposure target, darker or brighter", newline=True)
    auto_iso_limit: Field[int]  = Field(ISO_RANGE[1], min=ISO_RANGE[0], max=ISO_RANGE[1], widget=Widget.slider, description="Highest ISO auto exposure may use")
    anti_banding:   Field[AntiBanding] = Field(AntiBanding.MAINS_50_HZ, description="Mains flicker auto exposure avoids")
    exposure_region: Field[Rect] = Field(Rect(0.0, 0.0, 1.0, 1.0), newline=True, description="Part of the picture auto exposure measures, in fractions of it")
    balance:        Field[int]  = Field(BALANCE_RANGE[0], min=BALANCE_RANGE[0], max=BALANCE_RANGE[1], access=Field.READWRITE, widget=Widget.slider, description="White balance", newline=True)
    auto_balance:   Field[bool] = Field(True, widget=Widget.switch, description="Auto white balance")
    auto_balance_lock: Field[bool] = Field(False, widget=Widget.switch, description="Hold auto white balance where it is now")
    brightness:    Field[int]  = Field(0, min=BRIGHTNESS_RANGE[0], max=BRIGHTNESS_RANGE[1], widget=Widget.slider, description="Brightness", newline=True)
    contrast:       Field[int]  = Field(0, min=CONTRAST_RANGE[0], max=CONTRAST_RANGE[1], widget=Widget.slider, description="Contrast")
    saturation:     Field[int]  = Field(0, min=SATURATION_RANGE[0], max=SATURATION_RANGE[1], widget=Widget.slider, description="Saturation")
    denoise:        Field[int]  = Field(0, min=LUMA_DENOISE_RANGE[0], max=LUMA_DENOISE_RANGE[1], widget=Widget.slider, description="Luma denoise")
    chroma_denoise: Field[int]  = Field(0, min=CHROMA_DENOISE_RANGE[0], max=CHROMA_DENOISE_RANGE[1], widget=Widget.slider, description="Chroma denoise")
    sharpness:     Field[int]  = Field(0, min=SHARPNESS_RANGE[0], max=SHARPNESS_RANGE[1], widget=Widget.slider, description="Sharpness")
    dot_projector:  Field[float] = Field(0.0, min=0.0, max=1.0, step=0.05, widget=Widget.slider, description="IR dot projector brightness", newline=True)
    flood_light:    Field[float] = Field(0.0, min=0.0, max=1.0, step=0.05, widget=Widget.slider, description="IR flood light brightness")


class DepthSettings(BaseSettings):
    """Neural Depth's filters on the camera, live. A pixel that fails one reads 0, no depth."""
    confidence:     Field[int]   = Field(125, min=DEPTH_SCORE_RANGE[0], max=DEPTH_SCORE_RANGE[1], widget=Widget.slider, description="Drops depth the model is less sure of than this")
    edge:           Field[int]   = Field(10, min=DEPTH_SCORE_RANGE[0], max=DEPTH_SCORE_RANGE[1], widget=Widget.slider, description="Drops pixels with an edge magnitude below this")
    temporal_filter: Field[bool] = Field(False, widget=Widget.switch, description="Average depth over time", newline=True)
    temporal_alpha: Field[float] = Field(0.4, min=0.0, max=1.0, widget=Widget.slider, description="Weight of the new frame; lower smooths more")
    temporal_delta: Field[int]   = Field(100, min=DEPTH_SCORE_RANGE[0], max=DEPTH_SCORE_RANGE[1], widget=Widget.slider, description="Disparity change above which a pixel is not averaged")
    near:           Field[float] = Field(0.0, min=0.0, max=FAR_LIMIT_M, step=0.05, widget=Widget.slider, description="Drops depth nearer than this (m)", newline=True)
    far:            Field[float] = Field(FAR_LIMIT_M, min=0.0, max=FAR_LIMIT_M, step=0.05, widget=Widget.slider, description="Drops depth further than this (m); at the top, no limit")


class Oak4CameraSettings(RigCameraSettings):
    """One OAK4 camera: the contract's rate, mount and readings, plus what shapes its pipeline.

    An app shares `fps`, `color`, `resolution`, `model_path` and `fov` from its camera group, and the
    preset keys stay flat. `color` picks the delivered frame, color or grey, off the color sensor.
    """
    device_id:      Field[str]   = Field("", access=Field.INIT, description="Camera device ID (serial, or its IP; empty = the first OAK4 found)")
    link:           Field[CameraLink] = Field(CameraLink.USB, access=Field.INIT, description="Link to the PC; the other one is used when this one is missing")
    color:          Field[bool]  = Field(True, access=Field.INIT, newline=True, description="Color frames (False = grey off the same sensor)")
    model_chain:    Field[ModelChain] = Field(ModelChain.DETECTOR, access=Field.INIT, description="Models on the camera: none, a detector alone, one pose model, or a detector and a pose model on each crop")
    depth_model:    Field[DepthModel] = Field(DepthModel.OFF, access=Field.INIT, description="Neural Depth size on the camera; larger is sharper and slower")
    resolution:     Field[CameraResolution] = Field(CameraResolution.P800, access=Field.INIT, description="Delivered size on USB; the sensor mode follows it")
    ethernet_resolution: Field[CameraResolution] = Field(CameraResolution.P800, access=Field.INIT, description="Delivered size on Ethernet, which has less room")
    model_path:     Field[str]   = Field("data/models", access=Field.INIT)

    tilt:   Group[TiltCorrectionSettings] = Group(TiltCorrectionSettings)
    sensor: Group[SensorSettings] = Group(SensorSettings)
    depth:  Group[DepthSettings]  = Group(DepthSettings)
