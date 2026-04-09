from ..camera.definitions import FrameType, CoderType, CoderFormat
from modules.settings import BaseSettings, Field, Widget


class RecorderSettings(BaseSettings):

    # ── Initial settings ───────────────────────────────────────────────
    num_cameras:        Field[int]              = Field(1, access=Field.INIT, description="Number of cameras")
    fps:                Field[float]            = Field(30.0, access=Field.INIT, description="Camera frame rate")
    video_path:         Field[str]              = Field("recordings", description="Video recordings directory")
    temp_path:          Field[str]              = Field("temp", access=Field.INIT, description="Temporary files directory")
    video_encoder:      Field[CoderType]        = Field(CoderType.iGPU, access=Field.INIT, description="Video encoder type")
    video_format:       Field[CoderFormat]      = Field(CoderFormat.H264, access=Field.INIT, description="Video format")
    video_frame_types:  Field[list[FrameType]]  = Field([FrameType.VIDEO], access=Field.INIT, description="Frame types to record", visible=False)

    # ── Recording controls ────────────────────────────────────────────
    start:              Field[bool]             = Field(False, widget=Widget.button, description="Start recording", newline=True)
    stop:               Field[bool]             = Field(False, widget=Widget.button, description="Stop recording")
    recording:          Field[bool]             = Field(False, access=Field.READ, description="Recording active")
    group_id:           Field[str]              = Field("no_id", widget=Widget.input, description="Recording group ID")
    video_chunk_length: Field[float]            = Field(10.0, min=1.0, max=300.0, description="Video chunk duration (seconds)")
