from modules.cam.depthcam.Definitions import FrameType, CoderType, CoderFormat
from modules.settings import Settings, Field, Widget


class RecorderSettings(Settings):
    start:      Field[bool] = Field(False, widget=Widget.button, description="Start recording")
    stop:       Field[bool] = Field(False, widget=Widget.button, description="Stop recording")
    recording:  Field[bool] = Field(False, access=Field.READ, description="Recording active")
    group_id:   Field[str]  = Field("no_id", widget=Widget.input, description="Recording group ID")

    # Recorder-specific settings (moved from CameraSettings)
    video_encoder:      Field[CoderType]        = Field(CoderType.iGPU, access=Field.INIT, description="Video encoder type")
    video_chunk_length: Field[float]            = Field(10.0, min=1.0, max=300.0, description="Video chunk duration (seconds)")

    # Shared fields (populated from parent via share=[...] in Phase 4)
    video_path:         Field[str]              = Field("recordings", access=Field.INIT, visible=False, description="Video recordings directory")
    temp_path:          Field[str]              = Field("temp", access=Field.INIT, visible=False, description="Temporary files directory")
    video_format:       Field[CoderFormat]      = Field(CoderFormat.H264, access=Field.INIT, visible=False, description="Video format")
    video_frame_types:  Field[list[FrameType]]  = Field([FrameType.VIDEO], access=Field.INIT, visible=False, description="Frame types to record")
    color:              Field[bool]             = Field(True, access=Field.INIT, visible=False, description="Enable color capture")
    square:             Field[bool]             = Field(True, access=Field.INIT, visible=False, description="Use square aspect ratio")
    stereo:             Field[bool]             = Field(False, access=Field.INIT, visible=False, description="Enable stereo mode")
