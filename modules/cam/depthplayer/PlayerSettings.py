from modules.cam.depthcam.Definitions import FrameType, CoderType, CoderFormat
from modules.settings import Settings, Field, Widget


class PlayerSettings(Settings):
    available_folders: Field[list[str]] = Field(["–"], access=Field.READ, visible=False, description="Available recording folders")
    folder:         Field[str]  = Field("", widget=Widget.text_select, options=available_folders, description="Recording folder")
    start:          Field[bool] = Field(False, widget=Widget.button, description="Start playback")
    stop:           Field[bool] = Field(False, widget=Widget.button, description="Stop playback")
    current_chunk:  Field[int]  = Field(0, access=Field.READ, description="Current chunk")
    max_chunks:     Field[int]  = Field(0, access=Field.READ, description="Total chunks")
    range_start:    Field[int]  = Field(0, widget=Widget.number_field, description="Chunk range start")
    range_end:      Field[int]  = Field(0, widget=Widget.number_field, description="Chunk range end")

    # Player-specific settings (moved from CameraSettings)
    sim_fps:        Field[float]            = Field(30.0, min=1.0, max=120.0, access=Field.INIT, description="Simulation frame rate")
    sim_passthrough:Field[bool]             = Field(False, description="Simulation passthrough")
    video_decoder:  Field[CoderType]        = Field(CoderType.iGPU, access=Field.INIT, description="Video decoder type")

    # Shared fields (populated from parent via share=[...] in Phase 4)
    video_path:     Field[str]              = Field("recordings", access=Field.INIT, visible=False, description="Video recordings directory")
    video_format:   Field[CoderFormat]      = Field(CoderFormat.H264, access=Field.INIT, visible=False, description="Video format")
    video_frame_types: Field[list[FrameType]] = Field([FrameType.VIDEO], access=Field.INIT, visible=False, description="Frame types to record")
