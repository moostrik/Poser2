"""One Vimba camera's settings: the contract's `CameraSettings` plus the features of the camera
(Allied Vision Mako G-125C over GigE). Pure data; `SensorControl` writes the features to the camera."""

from .. import CameraSettings
from modules.settings import Field, Widget


class VimbaCameraSettings(CameraSettings):
    """The rate and the flips are live here: the camera takes a new frame rate at any time
    (`AcquisitionFrameRateAbs`) and the flips are done on the host, per frame.

    Exposure and gain each have auto, a manual value (sent while auto is off), a limit for auto
    and a readout of what the camera uses now; both autos aim for `target_brightness`, the
    exposure going up first and the gain after it. "Set WB" lets the camera measure its white
    balance once and takes the ratios it found into the panel.
    """
    flip_h:         Field[bool]  = Field(False, description="Mirror the camera image horizontally")
    flip_v:         Field[bool]  = Field(False, description="Flip the camera image vertically")
    fps:            Field[float] = Field(30.0, min=1.0, max=30.0, step=1.0, description="Frames per second asked of the camera; an exposure longer than a frame lowers it")

    # what the camera does now, whether auto or manual set it (the frame rate is `readings.video_fps`)
    dropped_frames:    Field[int]   = Field(0, access=Field.READ, newline=True, description="Frames that arrived incomplete and were skipped, since the start")
    exposure:          Field[float] = Field(0.0, access=Field.READ, description="Exposure time the camera uses now, in µs")
    gain:              Field[float] = Field(0.0, access=Field.READ, description="Gain the camera uses now, in dB")
    WB_red:            Field[float] = Field(0.0, access=Field.READ, description="White balance red ratio the camera uses now")
    WB_blue:           Field[float] = Field(0.0, access=Field.READ, description="White balance blue ratio the camera uses now")

    # exposure and gain: auto lengthens the exposure first (up to its max), then adds gain
    target_brightness: Field[int]   = Field(40, min=0, max=100, step=1, newline=True, description="Brightness that auto exposure and auto gain aim for (0-100)")

    manual_exposure:   Field[float] = Field(33000.0, min=12.0, max=50000.0, step=100.0, newline=True, description="Exposure time in µs (auto off)")
    auto_exposure:     Field[bool]  = Field(False, widget=Widget.switch, description="Automatic exposure time")
    exposure_max:      Field[float] = Field(33000.0, min=12.0, max=50000.0, step=100.0, description="Longest exposure time auto may use, in µs")

    manual_gain:       Field[float] = Field(0.0, min=0.0, max=30.0, step=0.1, newline=True, description="Gain in dB (auto off)")
    auto_gain:         Field[bool]  = Field(False, widget=Widget.switch, description="Automatic gain")
    gain_max:          Field[float] = Field(15.0, min=0.0, max=30.0, step=0.1, description="Most gain auto may add, in dB")

    manual_WB_red:     Field[float] = Field(1.0, min=0.5, max=4.0, step=0.01, newline=True, description="White balance red ratio (auto off)")
    manual_WB_blue:    Field[float] = Field(1.0, min=0.5, max=4.0, step=0.01, description="White balance blue ratio (auto off)")
    auto_WB:           Field[bool]  = Field(False, widget=Widget.switch, description="Continuous automatic white balance")
    set_WB:            Field[bool]  = Field(False, widget=Widget.button, description="Measure the white balance once and keep it (auto off)")

    saturation:    Field[float]     = Field(1.0, min=0.0, max=2.0, step=0.01, newline=True, description="Colour saturation (1 = unchanged)")
    hue:           Field[float]     = Field(0.0, min=-40.0, max=40.0, step=0.5, description="Hue rotation in degrees")
    gamma:         Field[float]     = Field(1.0, min=0.25, max=4.0, step=0.01, description="Gamma (1 = linear)")
