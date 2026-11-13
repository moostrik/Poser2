"""SMILE's camera settings: the Vimba camera's (`VimbaCameraSettings`, also read by the webcam for
its flips) with the app's sources around it, the image from `input/` and the device readout, and the
faces' settings: the upload, the detector and smile's face finder."""

from modules.camera.vimba import VimbaCameraSettings
from modules.inference import SerialUploaderSettings
from modules.inference.face import FaceDetectorSettings
from modules.settings import Field, Group, Widget

from ..face import FaceFinderSettings


class CameraSettings(VimbaCameraSettings):
    # the source: the camera device, or an image from input/ (whether a device was found or not)
    use_image:     Field[bool]      = Field(False, pinned=True, newline=True, description="Use the selected image instead of the camera")
    image_options: Field[list[str]] = Field([""], access=Field.READ, visible=False, description="Images in input/")
    image:         Field[str]       = Field("", widget=Widget.text_select, options=image_options, description="Image in input/ used as the source")
    image_count:   Field[int]       = Field(0, access=Field.READ, label="Images", description="Number of images in input/")
    reload:        Field[bool]      = Field(False, widget=Widget.button, description="Rescan input/ for images")
    image_next:    Field[bool]      = Field(False, label="Next after capture", description="Image source: move to the next image in input/ after each capture")

    device:        Field[str]       = Field("none", access=Field.READ, newline=True, description="Camera device in use")

    uploader:      Group[SerialUploaderSettings] = Group(SerialUploaderSettings)    # the frames to judge onto the GPU
    face_detector: Group[FaceDetectorSettings] = Group(FaceDetectorSettings)        # the faces and what they measure
    face:          Group[FaceFinderSettings] = Group(FaceFinderSettings)            # SMILE's choice of a subject and its crop, the image ComfyUI starts from
