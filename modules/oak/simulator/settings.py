from ..camera.definitions import FrameType, CoderType, CoderFormat
from modules.settings import BaseSettings, Field, Widget


class SimulatorSettings(BaseSettings):
    available_folders: Field[list[str]] = Field(["–"], access=Field.READ, visible=False, description="Available recording folders")
    folder:         Field[str]  = Field("", widget=Widget.text_select, options=available_folders, description="Recording folder")
    start:          Field[bool] = Field(False, widget=Widget.button, description="Start playback")
    stop:           Field[bool] = Field(False, widget=Widget.button, description="Stop playback")
    current_chunk:  Field[int]  = Field(0, access=Field.READ, description="Current chunk")
    max_chunks:     Field[int]  = Field(0, access=Field.READ, description="Total chunks")
    range_start:    Field[int]  = Field(0, widget=Widget.number_field, description="Chunk range start")
    range_end:      Field[int]  = Field(0, widget=Widget.number_field, description="Chunk range end")

    # Simulator-specific settings (moved from CameraSettings)
    sim_fps:        Field[float]            = Field(30.0, min=1.0, max=120.0, access=Field.INIT, description="Simulation frame rate")
    sim_passthrough:Field[bool]             = Field(False, description="Simulation passthrough")
    video_decoder:  Field[CoderType]        = Field(CoderType.iGPU, access=Field.INIT, description="Video decoder type")

    # Shared fields (populated from parent via share=[...])
    video_path:     Field[str]              = Field("recordings", access=Field.INIT, visible=False, description="Video recordings directory")
    video_format:   Field[CoderFormat]      = Field(CoderFormat.H264, access=Field.INIT, visible=False, description="Video format")
    video_frame_types: Field[list[FrameType]] = Field([FrameType.VIDEO], access=Field.INIT, visible=False, description="Frame types to record")
    num_cameras:    Field[int]              = Field(1, access=Field.INIT, visible=False, description="Number of cameras")
    color:          Field[bool]             = Field(True, access=Field.INIT, visible=False, description="Enable color capture")
    square:         Field[bool]             = Field(True, access=Field.INIT, visible=False, description="Use square aspect ratio")
    stereo:         Field[bool]             = Field(False, access=Field.INIT, visible=False, description="Enable stereo mode")
