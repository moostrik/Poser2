from modules.face.settings import FaceSettings
from modules.settings import BaseSettings, Field, Group, Widget


class CameraSettings(BaseSettings):
    # the source: the camera device, or an image from input/ (whether a device was found or not)
    use_image:     Field[bool]      = Field(False, pinned=True, newline=True, row_label="Source", label="Use image", description="Use the selected image instead of the camera")
    image_options: Field[list[str]] = Field([""], access=Field.READ, visible=False, description="Images in input/")
    image:         Field[str]       = Field("", widget=Widget.text_select, options=image_options, description="Image in input/ used as the source")
    image_count:   Field[int]       = Field(0, access=Field.READ, label="Images", description="Number of images in input/")
    reload:        Field[bool]      = Field(False, widget=Widget.button, description="Rescan input/ for images")

    device:        Field[str]       = Field("none", access=Field.READ, newline=True, row_label="Device", description="Camera device in use")
    flip_h:        Field[bool]      = Field(False, description="Mirror the camera image horizontally")
    flip_v:        Field[bool]      = Field(False, description="Flip the camera image vertically")

    gain_auto:     Field[bool]      = Field(False, widget=Widget.switch, newline=True, row_label="Gain", label="Auto", description="Vimba: automatic gain")
    gain_target:   Field[int]       = Field(40, min=0, max=100, step=1, label="Target", description="Vimba: auto gain target brightness (0-100)")

    face:          Group[FaceSettings] = Group(FaceSettings)     # SMILE's own face detection and crop, the image ComfyUI starts from
