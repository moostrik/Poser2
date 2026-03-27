from enum import Enum

from modules.cam.depthcam.Definitions import FrameType
from modules.settings import Settings, Field, Widget


class CoderType(Enum):
    CPU =   0
    GPU =   1
    iGPU =  2

class CoderFormat(Enum):
    H264 = '.mp4'
    H265 = '.hevc'


class PlayerSettings(Settings):
    folder:         Field[str]  = Field("", widget=Widget.input, description="Recording folder name")
    start:          Field[bool] = Field(False, widget=Widget.button, description="Start playback")
    stop:           Field[bool] = Field(False, widget=Widget.button, description="Stop playback")
    current_chunk:  Field[int]  = Field(0, access=Field.READ, description="Current chunk")
    max_chunks:     Field[int]  = Field(0, access=Field.READ, description="Total chunks")
    range_start:    Field[int]  = Field(0, widget=Widget.number_field, description="Chunk range start")
    range_end:      Field[int]  = Field(0, widget=Widget.number_field, description="Chunk range end")

class RecorderSettings(Settings):
    start:      Field[bool] = Field(False, widget=Widget.button, description="Start recording")
    stop:       Field[bool] = Field(False, widget=Widget.button, description="Stop recording")
    recording:  Field[bool] = Field(False, access=Field.READ, description="Recording active")
    group_id:   Field[str]  = Field("no_id", widget=Widget.input, description="Recording group ID")


class CameraSettings(Settings):

    # Camera device IDs
    ids:            Field[list[str]]        = Field([""], access=Field.INIT, description="Camera device IDs")
    num_cameras:    Field[int]              = Field(1, access=Field.READ, visible=False, description="Number of cameras (derived from ids)")

    # Core camera settings (require pipeline rebuild → INIT)
    fps:            Field[float]            = Field(30.0, min=1.0, max=120.0, access=Field.INIT, description="Camera frame rate")
    yolo:           Field[bool]             = Field(True, access=Field.INIT, description="Enable YOLO person detection")
    color:          Field[bool]             = Field(True, access=Field.INIT, description="Enable color capture")
    square:         Field[bool]             = Field(True, access=Field.INIT, description="Use square aspect ratio")
    stereo:         Field[bool]             = Field(False, access=Field.INIT, description="Enable stereo mode")
    hd_ready:       Field[bool]             = Field(False, access=Field.INIT, description="Use HD resolution")

    # Runtime-modifiable camera settings
    show_stereo:    Field[bool]             = Field(False, description="Show stereo visualization")
    manual:         Field[bool]             = Field(False, description="Manual camera control")
    flip_h:         Field[bool]             = Field(False, description="Flip horizontal")
    flip_v:         Field[bool]             = Field(False, description="Flip vertical")
    perspective:    Field[float]            = Field(0.0, min=-1.0, max=1.0, description="Perspective correction")

    # Paths (set once)
    model_path:     Field[str]              = Field("models", access=Field.INIT, visible=False, description="Model files directory")
    video_path:     Field[str]              = Field("recordings", access=Field.INIT, visible=False, description="Video recordings directory")
    temp_path:      Field[str]              = Field("temp", access=Field.INIT, visible=False, description="Temporary files directory")

    # Simulation
    sim_enabled:    Field[bool]             = Field(False, access=Field.INIT, description="Enable simulation mode")
    sim_passthrough:Field[bool]             = Field(False, description="Simulation passthrough")
    sim_fps:        Field[float]            = Field(30.0, min=1.0, max=120.0, access=Field.INIT, description="Simulation frame rate")

    # Recording
    video_chunk_length: Field[float]        = Field(10.0, min=1.0, max=300.0, description="Video chunk duration (seconds)")

    # Video encoding (set once)
    video_encoder:  Field[CoderType]        = Field(CoderType.iGPU, access=Field.INIT, description="Video encoder type")
    video_decoder:  Field[CoderType]        = Field(CoderType.iGPU, access=Field.INIT, description="Video decoder type")
    video_format:   Field[CoderFormat]      = Field(CoderFormat.H264, access=Field.INIT, description="Video format")
    video_frame_types: Field[list[FrameType]] = Field([FrameType.VIDEO], access=Field.INIT, description="Frame types to record")

    # Child settings for player and recorder
    player: PlayerSettings
    recorder: RecorderSettings
