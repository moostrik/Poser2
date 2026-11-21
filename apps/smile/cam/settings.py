"""SMILE's camera settings: the Vimba camera's (`VimbaCameraSettings`, also read by the webcam for
its flips) with the app's sources around it, the image from `input/` and the device readout, the OAK4's own (its
flips, rate and sensor, read with `-oak`), the faces' settings (the upload, the detector and smile's face finder), the
recording of the device's frames (the recorder and the session that splits it into chunks, started and stopped by one
pair of buttons), and, read only in -sim, the player of a recording and the replay camera it feeds."""

from modules.camera import CameraResolution, RigCameraSettings
from modules.camera.oak4 import ModelChain, Oak4CameraSettings
from modules.camera.vimba import VimbaCameraSettings
from modules.inference import SerialUploaderSettings
from modules.inference.face import FaceDetectorSettings
from modules.recording import Container, PlayerSettings, RecorderSettings
from modules.session import SessionSettings
from modules.settings import Field, Group, Widget

from ..face import FaceFinderSettings


class _RecorderSettings(RecorderSettings):
    """smile records the device's color as a clip any player plays, the metadata beside it: no depth, no mirror baked in."""
    container: Field[Container] = Field(Container.MP4, access=Field.INIT, visible=False, description="Chunk file: mp4 with the metadata beside")


class _Oak4Settings(Oak4CameraSettings):
    """smile's OAK4: color frames only, no model and no depth on the camera (smile finds the faces), at the sensor's
    4K mode, since the portrait is cut from the full frame. Tilt correction stays off: smile needs none. Its turn is
    how it is mounted, not the app's portrait mode: the view cuts the portrait out of an upright landscape frame."""
    model_chain: Field[ModelChain] = Field(ModelChain.OFF, access=Field.INIT, visible=False, description="Models on the camera: none")
    resolution:  Field[CameraResolution] = Field(CameraResolution.P2160, access=Field.INIT, description="Delivered size on USB; the sensor mode follows it")
    rotate90:    Field[bool] = Field(False, access=Field.INIT, description="The camera mounted on its side, turned a quarter")


class CameraSettings(VimbaCameraSettings):
    # the source: the camera device, or an image from input/ (whether a device was found or not)
    use_image:     Field[bool]      = Field(False, pinned=True, newline=True, description="Use the selected image instead of the camera")
    image_options: Field[list[str]] = Field([""], access=Field.READ, visible=False, description="Images in input/")
    image:         Field[str]       = Field("", widget=Widget.text_select, options=image_options, description="Image in input/ used as the source", access=Field.READWRITE)
    image_count:   Field[int]       = Field(0, access=Field.READ, label="Images", description="Number of images in input/")
    reload:        Field[bool]      = Field(False, widget=Widget.button, description="Rescan input/ for images")
    image_next:    Field[bool]      = Field(False, label="Next after capture", description="Image source: move to the next image in input/ after each capture")

    device:        Field[str]       = Field("none", access=Field.READ, newline=True, description="Camera device in use")

    start:         Field[bool]      = Field(False, widget=Widget.button, newline=True, description="Start recording the device's frames")
    stop:          Field[bool]      = Field(False, widget=Widget.button, description="Stop recording")
    split:         Field[bool]      = Field(False, widget=Widget.button, description="Split chunk", visible=False)
    split_seconds: Field[float]     = Field(60, min=1, max=60, widget=Widget.number, description="Split the recording into chunks of this length (s)")

    oak4:          Group[_Oak4Settings] = Group(_Oak4Settings)              # -oak: the OAK4, its own flips and rate
    uploader:      Group[SerialUploaderSettings] = Group(SerialUploaderSettings)    # the frames to judge onto the GPU
    face_detector: Group[FaceDetectorSettings] = Group(FaceDetectorSettings)        # the faces and what they measure
    face:          Group[FaceFinderSettings] = Group(FaceFinderSettings)            # SMILE's choice of a subject and its crop, the image ComfyUI starts from
    control:       Group[SessionSettings] = Group(SessionSettings, share=[start, stop, split, split_seconds])   # splits the recording into chunks
    recorder:      Group[_RecorderSettings] = Group(_RecorderSettings, share=[start, stop, split])             # the device's frames as they come
    simulator:     Group[PlayerSettings] = Group(PlayerSettings)          # -sim: the recording played in place of the device
    replay:        Group[RigCameraSettings] = Group(RigCameraSettings)    # -sim: the camera the recording's frames come from
