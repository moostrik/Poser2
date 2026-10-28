from modules.face.settings import FaceSettings
from modules.settings import BaseSettings, Field, Group, Widget


class CameraSettings(BaseSettings):
    # the source: the camera device, or an image from input/ (whether a device was found or not)
    use_image:     Field[bool]      = Field(False, pinned=True, newline=True, description="Use the selected image instead of the camera")
    image_options: Field[list[str]] = Field([""], access=Field.READ, visible=False, description="Images in input/")
    image:         Field[str]       = Field("", widget=Widget.text_select, options=image_options, description="Image in input/ used as the source")
    image_count:   Field[int]       = Field(0, access=Field.READ, label="Images", description="Number of images in input/")
    reload:        Field[bool]      = Field(False, widget=Widget.button, description="Rescan input/ for images")
    image_next:    Field[bool]      = Field(False, label="Next after capture", description="Image source: move to the next image in input/ after each capture")

    device:        Field[str]       = Field("none", access=Field.READ, newline=True, description="Camera device in use")
    flip_h:        Field[bool]      = Field(False, description="Mirror the camera image horizontally")
    flip_v:        Field[bool]      = Field(False, description="Flip the camera image vertically")
    target_frame_rate: Field[float] = Field(30.0, min=1.0, max=30.0, step=1.0, description="Vimba: frames per second asked of the camera; an exposure longer than a frame lowers it")

    # what the camera does now, whether auto or manual set it
    frame_rate:        Field[float] = Field(0.0, access=Field.READ, newline=True, description="Vimba: frames per second received from the camera")
    dropped_frames:    Field[int]   = Field(0, access=Field.READ, description="Vimba: frames that arrived incomplete and were skipped, since the start")
    exposure:          Field[float] = Field(0.0, access=Field.READ, description="Vimba: exposure time the camera uses now, in µs")
    gain:              Field[float] = Field(0.0, access=Field.READ, description="Vimba: gain the camera uses now, in dB")
    WB_red:            Field[float] = Field(0.0, access=Field.READ, description="Vimba: white balance red ratio the camera uses now")
    WB_blue:           Field[float] = Field(0.0, access=Field.READ, description="Vimba: white balance blue ratio the camera uses now")

    # exposure and gain: auto lengthens the exposure first (up to its max), then adds gain
    target_brightness: Field[int]   = Field(40, min=0, max=100, step=1, newline=True, description="Vimba: brightness that auto exposure and auto gain aim for (0-100)")

    manual_exposure:   Field[float] = Field(33000.0, min=12.0, max=50000.0, step=100.0, newline=True, description="Vimba: exposure time in µs (auto off)")
    auto_exposure:     Field[bool]  = Field(False, widget=Widget.switch, description="Vimba: automatic exposure time")
    exposure_max:      Field[float] = Field(33000.0, min=12.0, max=50000.0, step=100.0, description="Vimba: longest exposure time auto may use, in µs")

    manual_gain:       Field[float] = Field(0.0, min=0.0, max=30.0, step=0.1, newline=True, description="Vimba: gain in dB (auto off)")
    auto_gain:         Field[bool]  = Field(False, widget=Widget.switch, description="Vimba: automatic gain")
    gain_max:          Field[float] = Field(15.0, min=0.0, max=30.0, step=0.1, description="Vimba: most gain auto may add, in dB")

    manual_WB_red:     Field[float] = Field(1.0, min=0.5, max=4.0, step=0.01, newline=True, description="Vimba: white balance red ratio (auto off)")
    manual_WB_blue:    Field[float] = Field(1.0, min=0.5, max=4.0, step=0.01, description="Vimba: white balance blue ratio (auto off)")
    auto_WB:           Field[bool]  = Field(False, widget=Widget.switch, description="Vimba: continuous automatic white balance")
    set_WB:            Field[bool]  = Field(False, widget=Widget.button, description="Vimba: measure the white balance once and keep it (auto off)")

    saturation:    Field[float]     = Field(1.0, min=0.0, max=2.0, step=0.01, newline=True, description="Vimba: colour saturation (1 = unchanged)")
    hue:           Field[float]     = Field(0.0, min=-40.0, max=40.0, step=0.5, description="Vimba: hue rotation in degrees")
    gamma:         Field[float]     = Field(1.0, min=0.25, max=4.0, step=0.01, description="Vimba: gamma (1 = linear)")

    face:        Group[FaceSettings] = Group(FaceSettings)     # SMILE's own face detection and crop, the image ComfyUI starts from
