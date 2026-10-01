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

    gain_auto:     Field[bool]      = Field(False, widget=Widget.switch, newline=True, label="Auto gain", description="Vimba: automatic gain")
    gain_target:   Field[int]       = Field(40, min=0, max=100, step=1, description="Vimba: auto gain target brightness (0-100)")

    wb_auto:       Field[bool]      = Field(False, widget=Widget.switch, newline=True, label="Auto white balance", description="Vimba: continuous automatic white balance")
    wb_set:        Field[bool]      = Field(False, widget=Widget.button, label="Set white balance", description="Vimba: measure the white balance once and keep it (auto off)")
    wb_red:        Field[float]     = Field(1.0, min=0.5, max=4.0, step=0.01, label="Red", description="Vimba: white balance red ratio (auto off)")
    wb_blue:       Field[float]     = Field(1.0, min=0.5, max=4.0, step=0.01, label="Blue", description="Vimba: white balance blue ratio (auto off)")

    saturation:    Field[float]     = Field(1.0, min=0.0, max=2.0, step=0.01, newline=True, description="Vimba: colour saturation (1 = unchanged)")
    hue:           Field[float]     = Field(0.0, min=-40.0, max=40.0, step=0.5, description="Vimba: hue rotation in degrees")
    gamma:         Field[float]     = Field(1.0, min=0.25, max=4.0, step=0.01, description="Vimba: gamma (1 = linear)")

    face:        Group[FaceSettings] = Group(FaceSettings)     # SMILE's own face detection and crop, the image ComfyUI starts from
